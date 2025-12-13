import ply.lex as lex
import sys

class LexerClass:

    def __init__(self) -> None:
        self.lexer = lex.lex(module=self)
        

if __name__ == "__main__":
    print("buenas\n")
    lexer = LexerClass()
    if len(sys.argv) > 1:
        print("hola\n")
        lexer.test_with_files(sys.argv[1])