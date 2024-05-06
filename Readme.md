
## Inkrementalne uczenie się z odtwarzaniem wcześniejszych danych.

Użyte frameworki:
* Pytorch
* Pytorch Lightning
* Lucid library - https://github.com/greentfrapp/lucent = A library for visualisation of the neural networks. It generates an input image based on target.

Instalacja:
* Wywołaj skrypt install_libs.sh.
* W przypadku problemów wejdź w skrypt update_system.sh i spróbuj użyć którąś z komend. Uzycie na własną odpowiedzialność.
* Do uruchomienia wirtualnego środowiska wpisz w konsoli $source pythonEnv/bi/activate.
* Przy pierszym uruchomieniu można podać klucz wandb do dokładniejszego śledzenia logów.
* Dla lokalnego uruchomienia serwera wnadb, wpisz $wandb server start lub $python3 -m wandb server start.
* W przypadku problemów z ciągle działającym skryptem wandb po ubiciu procesu pythona, można wpisać komendę: $ ps aux|grep wandb|grep -v grep | awk '{print $2}'|xargs kill -9


Uruchomienie:
* W skrypcie main_experiments.py znajdują się na końcu pliku zakomentowane eksperymenty. Odkomentuj wybrane oraz uruchom opcjonalnie z flagami. Pomoc: $main_experiments.py -h
* Skrypt main.py służy do uruchomienia pojedynczego eksperymentu. Jako argumenty przyjmuje parametry tego eksperymentu. Pomoc: main.py -h
* Każdy ze skryptów można uruchomić z flagą -f do szybkiego sprawdzenia działania użytych innych komend.


Logi znajdują się w domyśłnych folderach './log_run/' oraz './model_save/'. Można je znaleźć poprzez hash lub w './model_save/' poprzez datę wykonania.
