def average_tokens(text):

    s = 0
    for t in text:
        s += len(t.split())

    return s / len(text)