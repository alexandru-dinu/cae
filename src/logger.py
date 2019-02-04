from termcolor import colored


def info(*msg):
    s = "[I] " + " ".join(list(map(str, msg)))
    print(colored(s, "green"))


def debug(*msg):
    s = "[D] " + " ".join(list(map(str, msg)))
    print(colored(s, "yellow"))


def error(*msg):
    s = "[E] " + " ".join(list(map(str, msg)))
    print(colored(s, "red"))
