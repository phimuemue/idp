import re

class Term:
    def __str__(self):
        return "[Some term]"

class Variable(Term):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name

class Number(Term):
    def __init__(self, number):
        self.number = number
    def __str__(self):
        return self.number

class Sum(Term):
    def __init__(self, *args):
        self.summands = args
    def __str__(self):
        return "+".join([str(x) for x in self.summands])

class Product(Term):
    def __init__(self, *args):
        self.factors = args
    def __str__(self):
        return "+".join([str(x) for x in self.factors])

class Minus(Term):
    def __init__(self, term):
        self.term = term
    def __str__(self):
        return "-%s"%(str(self.term))

class Quotient(Term):
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator
    def __str__(self):
        return "(%s)/(%s)"%(str(self.numerator), str(self.denominator))

print Sum(1,2,3)


def split_sum(s):
    s = re.sub(r"\++", "+", s)
    s = s.replace("-", "+-")
    if s[0]=="+":
        s = s[1:]
    count = 0
    splits = [-1]
    for (i, c) in enumerate(s):
        if c=="(":
            count = count + 1
        if c==")":
            count = count - 1
        if c=="+" and count==0:
            splits.append(i)
    splits.append(len(s))
    summand_str = [s[splits[i]+1:splits[i+1]] for i in xrange(len(splits)-1)]
    return [parse_string(x) for x in summand_str]

def parse_string(s):
    # trim outer brackets
    if s[0]=="(" and s[-1]==")":
        return parse_string(s[1:-1])
    # is string "negated"?
    if s[0]=="-":
        return Minus(parse_string(s[1:]))
    # look for first + or - sign

print [str(x) for x in split_sum("-a+b*(c+d-e)*g+a*c-f")]
