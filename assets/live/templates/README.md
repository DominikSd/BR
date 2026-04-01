To miejsce jest przeznaczone na proste assety heurystyczne dla adaptera `adapters/live`.

Struktura docelowa:
- `mobs/`
  Template packi dla 2 wariantow moba i ich prostych orientacji/obrotow.
- `occupied/`
  Template packi dla skrzyzowanych mieczykow nad zajeta grupka.

Na obecnym etapie perception pipeline potrafi dzialac bez tych assetow:
- w dry-run korzysta z kontrolowanych `template_hits`,
- w trybie analizy moze czytac pliki JSON z przygotowanymi hitami lub metadanymi.

To oznacza, ze ten katalog jest juz gotowym miejscem na kolejne iteracje:
- dodanie prawdziwego `template matching`,
- porownywanie skutecznosci roznych template packow,
- trzymanie referencyjnych assetow bez zmiany architektury.
