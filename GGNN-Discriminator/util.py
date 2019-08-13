import re
import string
import tensorflow as tf
import numpy as np

def prefix_sum(arr):
	res = [0]
	for a in arr: res.append(res[-1] + a)
	return res

# Identifier check function
c_sharp_keywords = ["abstract", "as", "base", "bool", "break", "byte", "case", "catch", "char", "checked", "class", "const", "continue", "decimal", "default", "delegate", "do", "double", "else", "enum", "event", "explicit", "extern", "false", "finally", "fixed", "float", "for", "foreach", "goto", "if", "implicit", "in", "int", "interface", "internal", "is", "lock", "long", "namespace", "new", "null", "object", "operator", "out", "override", "params", "private", "protected", "public", "readonly", "ref", "return", "sbyte", "sealed", "short", "sizeof", "stackalloc", "static", "string", "struct", "switch", "this", "throw", "true", "try", "typeof", "uint", "ulong", "unchecked", "unsafe", "ushort", "using", "using", "static", "virtual", "void", "volatile", "while", "add", "alias", "ascending", "async", "await", "descending", "dynamic", "from", "get", "global", "group", "into", "join", "let", "nameof", "orderby", "partial", "partial", "remove", "select", "set", "value", "var", "when", "where", "where", "yield"]
id_regex = "[a-zA-Z$_][a-zA-Z0-9$_]*"
def is_id(name):
	return re.match(id_regex, name) and name not in c_sharp_keywords

regex = re.compile('[%s]' % re.escape(string.punctuation))
def split_subtokens(token):
	def camel_case_split(identifier):
		if len(identifier) == 0: return []
		if not is_id(identifier):
			if identifier[0] == "\"": return ["\"str\""]
			elif identifier[0] == "'": return ["\'chr'"]
			else: return [c for c in identifier]
		if "_" in identifier:
			parts = identifier.split("_")
			return [m for p in parts for m in camel_case_split(p)]
		
		matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
		return [m.group(0).lower() for m in matches]
	
	if not is_id(token):
		if "\"" in token or "'" in token: return token
		elif token in c_sharp_keywords: return token
		elif re.match(regex, token): return token
		elif token == "<HOLE>": return token
		else: return tuple(token)
	parts = [p for t in token.split("_") for p in camel_case_split(t)]
	parts = [p.lower() for p in parts if len(p) > 0]
	if len(parts) == 0:
		parts = [token]
	return tuple(parts)