To miejsce jest przeznaczone na assety dla pixel-based MVP w `adapters/live`.

Kanoniczna struktura:
- `markers/`
  Opcjonalne sample i cropy czerwonych markerow. W obecnym etapie marker detector jest color-first,
  ale to miejsce zostaje na przyszle porownania i referencje.
- `mobs/mob_a/`
  Lokalny template pack dla pierwszego wariantu moba.
- `mobs/mob_b/`
  Lokalny template pack dla drugiego wariantu moba.
- `occupied/`
  Template pack dla skrzyzowanych mieczykow nad zajeta grupka.

Zasady tego etapu:
- pipeline jest marker-first:
  `sample frame -> ROI -> red marker detector -> occupied swords detector -> local mob confirmation -> merge/smoothing -> occupied/free -> nearest target`.
- czerwony marker jest wykrywany szybkim, color-based detektorem w ograniczonym ROI.
- template packi dla `mobs/` sa uzywane tylko do lokalnego potwierdzenia pod markerem,
  a nie do globalnego skanowania calego ekranu.
- w katalogach `mobs/` mozna trzymac:
  - bardziej ogolne template'y bazowe,
  - oraz `spot_*` / `spot_*_upper` cropy wyciete bezposrednio z realnych scen danego spota.
- obecny loader local confirmation preferuje template'y z `upper` w nazwie,
  bo daja ciasniejszy, szybszy i zwykle bardziej stabilny match pod markerem.
- pelniejsze cropy typu `base` / `spot_full` moga pozostac w katalogu jako material referencyjny
  do dalszego strojenia, ale nie musza byc aktywnie uzywane przez MVP.
- template pack dla `occupied/` jest uzywany lokalnie wokol kandydata do potwierdzenia zajetosci.
- occupied detection moze laczyc dwa sygnaly:
  - lokalny template pack skrzyzowanych mieczykow,
  - szybki color-first detector zielonych mieczykow w malym ROI nad markerem.
- dla mobow nadal mozna generowac proste warianty obrotu wedlug `live.template_rotations_deg`.

To jest swiadomie MVP:
- bez OCR,
- bez ciezkiego CV/ML,
- bez przebudowy `domain` i `application`.

W kolejnych etapach ten sam katalog moze przechowywac:
- sample czerwonych markerow dla roznych spotow,
- dodatkowe template packi,
- lepiej przyciete warianty,
- osobne profile dla roznych spotow lub kamer.
