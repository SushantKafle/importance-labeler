def align_text(text_1, text_2, display=False):
    mis_alignments = []
    text_1 = text_1.split(" ")
    text_2 = text_2.split(" ")

    l_matrix = get_levenstein_matrix(text_1, text_2)
    paths = traverse_matrix(l_matrix, text_1, text_2)

    ins_err_cnt, del_err_cnt, sub_err_cnt = (0, 0, 0)
    for path in paths:
        ref_id = -1
        if path[2] == 'I':
            ins_err_cnt += 1
            ref_word = ''
            hyp_word = text_2[path[1]]
        elif path[2] == 'D':
            del_err_cnt += 1
            hyp_word = ''
            ref_word = text_1[path[0]]
            ref_id = path[0]
        elif path[2] == 'S':
            sub_err_cnt += 1
            hyp_word = text_2[path[1]]
            ref_word = text_1[path[0]]
            ref_id = path[0]
        else:
            hyp_word = text_2[path[1]]
            ref_word = text_1[path[0]]
            ref_id = path[0]

        if display:
            print (ref_word + " ==> " + hyp_word)

        mis_alignments.append({'R': ref_word, 'H': hyp_word, 'ID': ref_id})

    if display:
        print ("Summary: \n Total errors: %d \n Substitution error(%d), Deletion error (%d), Insertion error (%d) \n" \
            % (ins_err_cnt + del_err_cnt + sub_err_cnt, sub_err_cnt, del_err_cnt, ins_err_cnt))

    return mis_alignments, {'I': ins_err_cnt, 'S': sub_err_cnt, 'D': del_err_cnt}


def get_levenstein_matrix(reference, hypothesis):
    l_matrix = [[0 for x in range(len(hypothesis) + 1)] for y in range(len(reference) + 1)]
    for i in range(len(reference)):
        for j in range(len(hypothesis)):
            if i == 0:
                l_matrix[0][j] = j
            elif j == 0:
                l_matrix[i][0] = i

    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i-1] == hypothesis[j-1]:
                l_matrix[i][j] = l_matrix[i-1][j-1]
            else:
                cost_substitution = l_matrix[i-1][j-1] + (1 + abs(len(reference[i-1]) - len(hypothesis[j-1])))
                cost_insertion = l_matrix[i][j-1] + (len(hypothesis[j-1]) + 1)
                cost_deletion = l_matrix[i-1][j] + (len(reference[i-1]) + 1)
                l_matrix[i][j] = min(cost_substitution, cost_insertion, cost_deletion)

    return l_matrix


def traverse_matrix(matrix, reference, hypothesis):
    paths = []
    lr = len(reference)
    lh = len(hypothesis)
    t = 'S'
    while (True):

        if reference[lr - 1] == hypothesis[lh - 1]:
            lr = lr - 1
            lh = lh - 1
            paths.append([lr, lh, ''])
        else:
            a = [matrix[lr - 1][lh], matrix[lr][lh - 1], matrix[lr - 1][lh - 1]]
            i = a.index(min(a))
            if i == 0:
                lr = lr - 1
                t = 'D'
            elif i == 1:
                lh = lh - 1
                t = 'I'
            else:
                lr = lr - 1
                lh = lh - 1
                t = 'S'
            paths.append([lr, lh, t])
        if lr < 1 and lh < 1:
            break
    return list(reversed(paths))


if __name__ == "__main__":
    import sys
    arguments = sys.argv

    if len(arguments) < 3:
        print ("Not enough arguments: python align_text.py <arg1> <arg2>")
    else:
        print (align_text(arguments[1], arguments[2]))

