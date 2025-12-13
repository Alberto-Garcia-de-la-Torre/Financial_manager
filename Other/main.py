from ajs_lexer import LexerClass
import sys

# lexer = LexerClass()

if __name__ == '__main__':
    html_filename = sys.argv[1]
    file = open(html_filename, 'r')
    print(file)
    content = file.read()
    print(content[:5])