import ply.lex as lex
import sys

class LexerClass:
    tokens = (
        'OL_COMMENT',
        'ML_COMMENT',
        'TRUE',
        'FALSE',
        'LET',
        'INT',
        'FLOAT',
        'CHARACTER',
        'WHILE',
        'BOOLEAN',
        'FUNCTION',
        'RETURN',
        'TYPE',
        'IF',
        'ELSE',
        'NULL',
        'DECIMAL',
        'BINARIO',
        'OCTAL',
        'HEXADECIMAL',
        'PUNTODECIMAL',
        'CIENTIFICA'
    )

    def __init__(self) -> None:
        self.lexer = lex.lex(module=self)
    
    # Expresiones regulares para tokens simples
    t_TRUE = r'tr'
    t_FALSE = r'fl'
    t_LET = r'let'
    t_INT = r'int'
    t_FLOAT = r'float'
    t_CHARACTER = r'character'
    t_WHILE = r'while'
    t_BOOLEAN = r'boolean'
    t_FUNCTION = r'function'
    t_RETURN = r'return'
    t_TYPE = r'type'
    t_IF = r'if'
    t_ELSE = r'else'
    t_NULL = r'null'


    # Expresiones regulares para tokens complejos
    def t_OL_COMMENT(self, t):
        r'\/\/.*'
        return t
    
    def t_ML_COMMENT(self, t):
        r'\/\*[.*\n]\*\/'
        return t
    
    def t_DECIMAL(self, t):
        r'-?[1-9][0-9]*|0$'                     # Comprobar que está bien hecho
        return t
    
    def t_BINARIO(self, t):
        r'0[bB][0-9]+'
        return t
    
    def t_OCTAL(self, t):
        r'0[1-9][0-9]+'                         # Puede que deba ir delante del decimal
        return t
    
    def t_HEXADECIMAL(self, t):
        r'0[xX][0-9a-fA-F]+'                    # Preguntar si solo debe haber 4 números detras de la x
        return t
    
    def t_PUNTODECIMAL(self, t):
        r'[-?[1-9][0-9]*|0]?.[[0-9]*[1-9]]?'    # Creo que este está mal
        return t

    def t_CIENTIFICA(self, t):
        r''                                     # Hacer cuando sepa que puntodecimal está bien hecho
        return t    

    # Manejo de errores de token
    def t_error(self, t):
        print(f"Carácter no válido: '{t.value[0]}' en la línea {t.lineno}")
        t.lexer.skip(1)

    # Ignorar caracteres como tabulaciones y saltos de línea
    t_ignore = ' \t'



    def test(self, data):
        self.lexer.input(data)
        for token in self.lexer:
            print(token.type, token.value)
    
    def test_with_files(self, path):
        file = open(path)
        content = file.readlines()
        for line in content:
            self.test(line)
        

if __name__ == "__main__":
    print("buenas\n")
    lexer = LexerClass()
    if len(sys.argv) > 1:
        print("hola\n")
        lexer.test_with_files(sys.argv[1])