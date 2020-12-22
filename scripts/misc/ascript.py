pp = [.826,.868]
bc = self.barcodes[-1]
dio_dat = bc.get_dio_dat()
dat = dio_dat['res'][-1]
Q = bc.query(dat, pp[0], pp[1])

bs = np.array(Q['birth'][1])
ds = np.array(Q['death'][1])
cl = np.array(Q['chain'][1])


# ax.imshow(bc.G.T)
G = np.array([[a if a >= 0.8 else 0 for a in r] for r in bc.G]).T
ax.imshow(G)

ax.plot(bs[:,0], bs[:,1], c='green', zorder=1)

_ds = np.vstack((ds,ds[0]))
ax.plot(_ds[:,0], _ds[:,1], c='red', zorder=2)

for e in cl:
    ax.plot(e[:,0], e[:,1], c='blue', zorder=3)





def get_birth(d, pt):
    return d['filt'][pt.data]
def get_death(d, pt):
    return d['filt'][d['hom'].pair(pt.data)]

df = [pt for pt in dat['full']['dgms'][1] if pt.birth >= self.cuts[0]]
dr = dat['res'][0]['dgms'][1]

df_dat = dat['full']
dr_dat = dat['res'][0]

dfbd = [(pt, get_birth(df_dat, pt), get_death(df_dat, pt)) for pt in df if pt.death < np.inf]
drbd = [(pt, get_birth(dr_dat, pt), get_death(dr_dat, pt)) for pt in dr if pt.death < np.inf]

fdict, rdict = {tuple(d) : (p, b, d) for p,b,d in dfbd}, {tuple(d) : (p, b, d) for p,b,d in drbd}

bad_matches = [(f, rdict[fd]) for fd, f in fdict.items() if fd in rdict and rdict[fd][1] != f[1]]
no_matches = [f for fd, f in fdict.items() if not fd in rdict]
