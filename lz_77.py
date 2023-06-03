def find_matching(search, lookahead):
    best_length = 0
    best_position = -1

    for i in range(len(search)):
        length = 0

        while length < len(lookahead) and length + i < len(search) and search[i + length] == lookahead[length]:
            length += 1

        if best_length < length:
            best_length = length
            best_position = len(search) - i

    return best_position, best_length


def lz77_encode(input_string, search_len=20):
    ans = []
    pos = 0
    search = ''
    lookahead = input_string

    while pos < len(input_string):
        offset, length = find_matching(search, lookahead)
        if offset == -1:
            ans.append(input_string[pos])
            pos += 1
        else:
            ans.append("<{},{}>".format(offset, length))
            pos += length

        search = input_string[max(0, pos - search_len):pos]
        lookahead = input_string[pos:]

    return "".join(ans)
