from lark import Lark, UnexpectedCharacters
from utils.cfg.ast import Node
from utils.cfg.naming import NamingError
from utils.cfg.ast import FOLTransformer

class CFGParser:
    def __init__(self):
        with open('utils/cfg/syntax.lark', 'r') as file:
            grammar = file.read()
        self.parser = Lark(grammar, parser='earley')

    def parse(self, text: str) -> Node:
        try:
            tree = self.parser.parse(text)
            return FOLTransformer().transform(tree)
        except UnexpectedCharacters as e:
            raise NamingError(self.parser, e, text)

