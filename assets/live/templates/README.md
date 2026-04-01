To miejsce jest przeznaczone na assety dla pixel-based MVP w `adapters/live`.

Kanoniczna struktura:
- `mobs/mob_a/`
  Template pack dla pierwszego wariantu moba.
- `mobs/mob_b/`
  Template pack dla drugiego wariantu moba.
- `occupied/`
  Template pack dla skrzyzowanych mieczykow nad zajeta grupka.

Zasady tego etapu:
- bazowy matcher laduje wszystkie pliki `.png` z tych katalogow,
- dla mobow generuje proste warianty obrotu wedlug `live.template_rotations_deg`,
- dla mieczykow uzywa wariantow zapisanych bezposrednio w katalogu `occupied/`,
- pipeline pozostaje prosty i w pelni heurystyczny:
  `sample frame -> ROI -> template matching -> merge hits -> occupied/free -> nearest target`.

To jest swiadomie MVP:
- bez OCR,
- bez ciezkiego CV/ML,
- bez przebudowy `domain` i `application`.

W kolejnych etapach ten sam katalog moze przechowywac:
- dodatkowe template packi,
- lepiej przyciete warianty,
- osobne profile dla roznych spotow lub kamer.
