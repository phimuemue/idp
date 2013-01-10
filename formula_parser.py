import re

class Term:
    def __str__(self):
        return "[Some term]"
    def simplify(self):
        return self

class Variable(Term):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name

class Number(Term):
    def __init__(self, number):
        self.number = number
    def __str__(self):
        return str(self.number)

class Sum(Term):
    def __init__(self, *args):
        self.summands = args
    def __str__(self):
        return "+".join(["(%s)"%str(x) for x in self.summands])
    def simplify(self):
        if len(self.summands) == 1:
            return self.summands[0].simplify()
        return self

class Product(Term):
    def __init__(self, *args):
        self.factors = args
    def __str__(self):
        return "*".join(["(%s)"%str(x) for x in self.factors])

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

class Power(Term):
    def __init__(self, base, exponent):
        self.base = base
        self.exponent = exponent
    def __str__(self):
        return "(%s)**(%s)"%(str(self.base), str(self.exponent))

def split_sum_and_prod(s):
    s = re.sub(r"\++", "+", s)
    s = re.sub(r"\+-", "-", s)
    s = s.replace("-", "+-")
    if s[0]=="+":
        s = s[1:]
    # first, we try for a sum
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
    if len(summand_str) > 1:
        print summand_str
        return Sum(*[parse_string(x) for x in summand_str])
    # if we come here, we have no sum, so we try product
    print "No sum found."
    count = 0
    splits = [-1]
    for (i, c) in enumerate(s):
        if c=="(":
            count = count + 1
        if c==")":
            count = count - 1
        if c=="*" and count==0:
            splits.append(i)
    splits.append(len(s))
    factors_str = [s[splits[i]+1:splits[i+1]] for i in xrange(len(splits)-1)]
    if len(factors_str) > 1:
        print factors_str
        return Product(*[parse_string(x) for x in factors_str])
    print "No product found."
    count = 0
    splits = [-1]
    for (i, c) in enumerate(s):
        if c=="(":
            count = count + 1
        if c==")":
            count = count - 1
        if c=="^" and count==0:
            splits.append(i)
    assert(len(splits)<=2)
    splits.append(len(s))
    if len(splits)==3:
        return Power(parse_string(s[:splits[1]]), parse_string(s[splits[1]+1]))
    print factors_str
    return Product(*[parse_string(x) for x in factors_str])

def parse_string(s):
    raw_input ("Parsing %s"%s)
    # trim outer brackets
    if s[0]=="(" and s[-1]==")":
        return parse_string(s[1:-1])
    # is string "negated"?
    if s[0]=="-":
        return Minus(parse_string(s[1:]))
    # is it a simple number?
    if re.search(r"^-?\d+(?:\.d+)?(?:[eE]d+)?$", s) is not None:
        return Number(float(s))
    # is it a simple variable?
    if re.search(r"^([a-zA-Z_]\d*[a-zA-Z_]*)$", s) is not None:
        return Variable(s)
    # probably something else
    return split_sum_and_prod(s)
    

print parse_string("2^a-c")
print parse_string("a+(2*b-c)+(2+b*2)-2")
