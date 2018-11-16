import sys



idx = 2
for line in sys.stdin:
    ts = line.strip().split(' ')
    if len(ts) < 5:
        continue
    gid = int(ts[0].split(':')[0])
    print '%d: f%d'%(idx, gid), ' #', ' '.join(ts[1:])
    idx += 1

for line in sys.stdin:
    ts = line.strip().split(' ')
    if len(ts) < 5:
        continue
    gid = int(ts[0].split(':')[0])
    bucket = int(ts[0].split(':')[-1])
    print '## ', ' '.join([t for t in ts[2:] if len(t) > 0])
    print 'f%d:'%gid     
    print '  type: category'
    print '  transform: hash_bucket'
    print '  parameter: %d'%bucket
    print ''

