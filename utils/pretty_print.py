from dataclasses import dataclass, field
from colorama import Fore, Back, Style

@dataclass
class COLOR():
    NORMAL: str = Fore.BLUE
    NORMAL_2: str = Fore.CYAN
    NORMAL_3: str = Fore.GREEN
    NORMAL_4: str = Fore.LIGHTMAGENTA_EX
    DETAIL: str = Fore.YELLOW
    WARNING: str = Fore.RED
    RESET: str = Style.RESET_ALL

def sprint(string, *args):
    print(string, *args, f"{Style.RESET_ALL}")