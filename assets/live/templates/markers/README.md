To miejsce jest przeznaczone na sample i cropy markerow nad targetami.

W obecnym etapie:
- detekcja markerow jest nadal color-first,
- ale katalog `markers` jest juz aktywnie uzywany jako lekki fallback template-based,
- cropy stad pomagaja wtedy, gdy sam kolor nie daje `candidate_hits`, mimo ze marker jest widoczny na klatce.

Mozna tu przechowywac:
- male cropy roznych wariantow zoltych albo czerwonych markerow,
- sample z roznych spotow i ustawien kamery,
- material do szybkiego strojenia marker stage bez ruszania calego pipeline'u.

Wskazowki praktyczne:
- wrzucaj tylko same markery albo bardzo male cropy wokol markera,
- nie wrzucaj tu pelnych sylwetek mobow ani graczy,
- nazwy typu `yellow_marker_01.png` sa wystarczajace.
