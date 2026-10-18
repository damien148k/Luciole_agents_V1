"""Watcher : garde l'index aligné sur une racine de dossiers, sans intervention.

V5 empilait un observateur ``watchdog``, des minuteries par fichier, une file
de travaux SQLite et un réconciliateur ; la panne du 3 août 2026 (cinq
suppressions simultanées, connexion SQLite partagée entre fils, plus aucune
écriture jusqu'au redémarrage) venait de cet empilement. Ici, un seul fil :

1. toutes les ``interval`` secondes, un **relevé** de la racine (chemin,
   taille, date de modification des fichiers que le pipeline lirait) ;
2. si le relevé diffère du dernier relevé indexé et qu'il est **stable**
   (identique au relevé précédent : plus aucun fichier en cours de copie),
   un passage du pipeline, qui est lui-même le réconciliateur (nouveaux,
   modifiés, supprimés) et ne réanalyse pas les documents inchangés ;
3. un passage en échec n'arrête pas le watcher : il est retenté plus tard,
   avec une attente qui double à chaque échec (plafonnée). L'index étant écrit
   avant le magasin, un passage interrompu est simplement refait.

Le relevé par interrogation fonctionne partout, y compris sur un dossier
Windows monté dans Docker, où les notifications du système de fichiers ne
traversent pas. Un verrou de fichier empêche deux watchers sur le même
magasin ; le système le libère si le processus meurt.

    python -m luciole.index.watcher <racine> --profile ventes --hardware cpu
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from luciole.core.profile import UseCaseProfile
from luciole.ingestion.dossiers import Registry, is_ignored

Snapshot = dict[str, tuple[int, int]]


def snapshot(root: str | Path, profile: UseCaseProfile, registry: Registry) -> Snapshot:
    """Relevé des fichiers que le pipeline lirait : chemin relatif → (taille,
    date de modification en ns). Les fichiers temporaires (« ~$ » de Word,
    « .~lock » de LibreOffice, fichiers cachés) et les formats non pris en
    charge n'y figurent pas : ils ne déclenchent aucun passage."""
    root = Path(root)
    out: Snapshot = {}
    if not root.is_dir():
        return out
    dossier_file = profile.ingestion.dossier_file
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if len(rel.parts) < 2 or any(is_ignored(Path(p)) for p in rel.parts):
            continue  # hors de tout dossier, ou ignoré
        try:
            if not path.is_file():
                continue
            if not (rel.name == dossier_file and len(rel.parts) == 2) and not registry.supports(
                path
            ):
                continue
            stat = path.stat()
        except OSError:
            continue  # disparu entre le parcours et la lecture : vu au relevé suivant
        out[rel.as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return out


@dataclass
class WatchSettings:
    interval: float = 10.0  # secondes entre deux relevés
    max_backoff: float = 600.0  # attente maximale après des échecs répétés


class Watcher:
    """Boucle à un seul fil. ``run`` lance un passage du pipeline ; ``take``
    prend un relevé. ``clock`` et ``sleep`` sont remplaçables dans les tests."""

    def __init__(
        self,
        run: Callable[[], object],
        take: Callable[[], Snapshot],
        settings: WatchSettings | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.run = run
        self.take = take
        self.settings = settings or WatchSettings()
        self.clock = clock
        self.sleep = sleep
        self.indexed: Snapshot | None = None  # relevé du dernier passage réussi
        self.previous: Snapshot | None = None  # relevé de l'intervalle précédent
        self.failures = 0
        self.retry_at = 0.0
        self.passes = 0
        self.stopped = False

    def tick(self) -> bool:
        """Un relevé ; lance un passage si besoin. Renvoie True si un passage
        a été lancé (réussi ou non)."""
        current = self.take()
        stable = current == self.previous
        self.previous = current
        if current == self.indexed:
            return False
        if self.indexed is not None and not stable:
            return False  # fichiers en cours de copie : attendre un relevé identique
        if self.clock() < self.retry_at:
            return False
        self.passes += 1
        try:
            self.run()
        except Exception as e:  # noqa: BLE001 - le watcher ne s'arrête pas sur un échec
            self.failures += 1
            wait = min(self.settings.interval * 2**self.failures, self.settings.max_backoff)
            self.retry_at = self.clock() + wait
            logger.error(f"passage en échec ({self.failures} de suite), nouvel essai dans "
                         f"{wait:.0f} s : {e}")  # fmt: skip
            return True
        if self.failures:
            logger.info(f"passage réussi après {self.failures} échec(s)")
        self.failures = 0
        self.retry_at = 0.0
        # Le relevé pris avant le passage : un fichier modifié pendant le
        # passage diffère de lui et sera repris au relevé suivant.
        self.indexed = current
        return True

    def loop(self, max_ticks: int | None = None) -> None:
        """Relevé, puis attente, jusqu'à ``stop()`` (ou ``max_ticks`` relevés)."""
        ticks = 0
        while not self.stopped and (max_ticks is None or ticks < max_ticks):
            self.tick()
            ticks += 1
            if not self.stopped and (max_ticks is None or ticks < max_ticks):
                self.sleep(self.settings.interval)

    def stop(self) -> None:
        self.stopped = True


class WatcherLock:
    """Verrou exclusif sur ``<magasin>.watch.lock`` : un seul watcher par
    magasin. Le verrou est tenu par le système (``msvcrt`` sous Windows,
    ``fcntl`` ailleurs) et disparaît avec le processus, même tué."""

    def __init__(self, store_path: str | Path) -> None:
        self.path = Path(f"{store_path}.watch.lock")
        self._file = None

    def acquire(self) -> WatcherLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+")  # noqa: SIM115 - gardé ouvert tant que le verrou vit
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            handle.close()
            raise RuntimeError(
                f"un autre watcher surveille déjà ce magasin ({self.path}) : "
                "un seul watcher par cas d'usage"
            ) from e
        self._file = handle
        return self

    def release(self) -> None:
        if self._file is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None

    def __enter__(self) -> WatcherLock:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()


def main(argv: list[str] | None = None, *, max_ticks: int | None = None, **services) -> int:
    """Mêmes options que ``python -m luciole.index.pipeline``, plus
    ``--interval`` ; ``services`` (embedder_factory, qdrant, opensearch)
    remplace le vrai modèle et les vrais services dans les tests."""
    from luciole.index.pipeline import build, parser

    cli = parser("python -m luciole.index.watcher", "Surveille une racine de dossiers "
                 "et garde l'index à jour (Ctrl+C pour arrêter).")  # fmt: skip
    cli.add_argument("--interval", type=float, help="secondes entre deux relevés (défaut : profil)")
    args = cli.parse_args(argv)
    with build(args, **services) as ctx:
        settings = WatchSettings(interval=args.interval or ctx.profile.ingestion.watch_interval_s)
        with WatcherLock(ctx.store_path):
            watcher = Watcher(
                lambda: ctx.pipeline.run(args.root),
                lambda: snapshot(args.root, ctx.profile, ctx.pipeline.registry),
                settings,
            )
            print(f"Surveillance de {args.root} toutes les {settings.interval:g} s")
            try:
                watcher.loop(max_ticks)
            except KeyboardInterrupt:
                print("Arrêt demandé.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
