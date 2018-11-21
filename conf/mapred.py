import sys

def read_conf(fname):
    order = {}
    f = open(fname)
    for line in f:
        if line[0] == '#' or len(line) < 2:
            continue
        ts = line.strip().split(' ')
        if len(ts) < 2:
            continue
        idx = int(ts[0].split(':')[0])
        if idx <= 1:
            continue
        for t in ts[1:]:
            if len(t) > 0:
                gid = t
                order[idx] = gid
                break
    return order

max_sequence_len = 20
sequence_ids = {'f20164':100000, 'f20165':100000}

if __name__ == '__main__':
    order = read_conf(sys.argv[1])
    for line in sys.stdin:
        fd = {}
        ts = line.strip().split('\t')
        output = [ts[0]]
        for t in ts[1:]:
            gid = 'f%d'%(int(t) >> 48)
            if gid not in fd:
                fd[gid] = [t]
            else:
                fd[gid].append(t)

        for sid in sequence_ids:
            hash_space = sequence_ids[sid]
            slen = 0
            if sid in fd:
                slen = len(fd[sid])
            else:
                fd[sid] = []
            if slen > 0:
                raw_seq = fd[sid]
                fd[sid] = []
                for i in range(slen):
                    fd[sid].append('%d'%(int(raw_seq[i]) % hash_space))
            for i in range(max_sequence_len - slen):
                fd[sid].append('0')
            fd[sid + 'len'] = [str(slen)]

        for i in range(2,2+len(order)):
            gid = order[i]
            if gid not in fd:
                output.append('')
            else:
                output.append(','.join(fd[gid]))
        print '\t'.join(output)

