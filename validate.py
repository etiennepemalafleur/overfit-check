# Reference implementation that EXACTLY mirrors the JS (mulberry32 PRNG + algorithms),
# so the browser's numbers can be validated against these.
import math
from itertools import combinations
GAMMA=0.5772156649015329
def make_rng(seed):
    state={'a':seed & 0xffffffff}
    def imul(x,y): return ((x & 0xffffffff)*(y & 0xffffffff)) & 0xffffffff
    def rnd():
        state['a']=(state['a']+0x6D2B79F5)&0xffffffff
        a=state['a']
        t=imul(a ^ (a>>15), 1|a)
        t=(t + imul(t ^ (t>>7), 61|t))&0xffffffff
        t=t ^ (t>>14)
        return (t & 0xffffffff)/4294967296.0
    return rnd
def gauss(rnd):
    u1=rnd() or 1e-12; u2=rnd()
    return math.sqrt(-2*math.log(u1))*math.cos(2*math.pi*u2)
def gen(seed,T,N,means,sigma=0.01):
    rnd=make_rng(seed); M=[[0.0]*N for _ in range(T)]
    for t in range(T):
        for i in range(N):
            M[t][i]=means[i]+gauss(rnd)*sigma
    return M
# ---- stats ----
def col_sr(M):
    T=len(M);N=len(M[0]);out=[]
    for i in range(N):
        s=sum(M[t][i] for t in range(T));mu=s/T
        v=sum((M[t][i]-mu)**2 for t in range(T))/T
        out.append(mu/math.sqrt(v) if v>0 else 0.0)
    return out
def norm_cdf(x): return 0.5*math.erfc(-x/math.sqrt(2))
def norm_ppf(p):
    # Acklam
    a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00]
    b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01]
    c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00]
    d=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00]
    pl=0.02425
    if p<pl:
        q=math.sqrt(-2*math.log(p));return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p<=1-pl:
        q=p-0.5;r=q*q;return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q=math.sqrt(-2*math.log(1-p));return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
def emax(N): return (1-GAMMA)*norm_ppf(1-1.0/N)+GAMMA*norm_ppf(1-1.0/(N*math.e))
def moments(col):
    n=len(col);mu=sum(col)/n;m2=sum((x-mu)**2 for x in col)/n;sd=math.sqrt(m2)
    g3=sum((x-mu)**3 for x in col)/n/sd**3; g4=sum((x-mu)**4 for x in col)/n/sd**4
    return mu/sd,g3,g4,n
def dsr(M):
    sr=col_sr(M);N=len(sr);j=max(range(N),key=lambda i:sr[i])
    col=[M[t][j] for t in range(len(M))]
    SR,g3,g4,n=moments(col)
    mean_sr=sum(sr)/N;var_sr=sum((s-mean_sr)**2 for s in sr)/(N-1)
    SR0=math.sqrt(var_sr)*emax(N)
    psr=norm_cdf((SR-SR0)*math.sqrt(n-1)/math.sqrt(1-g3*SR+(g4-1)/4*SR*SR))
    return psr,SR*math.sqrt(252),SR0*math.sqrt(252),j
def array_split(T,S):
    base=T//S;rem=T%S;idx=[];start=0
    for i in range(S):
        sz=base+(1 if i<rem else 0);idx.append(list(range(start,start+sz)));start+=sz
    return idx
def cscv(M,S=16):
    T=len(M);N=len(M[0]);T2=(T//S)*S;blocks=array_split(T2,S)
    bsum=[[0.0]*N for _ in range(S)];bsq=[[0.0]*N for _ in range(S)];bcnt=[0]*S
    for bi,b in enumerate(blocks):
        bcnt[bi]=len(b)
        for t in b:
            for i in range(N):
                bsum[bi][i]+=M[t][i];bsq[bi][i]+=M[t][i]**2
    def sr(sel):
        out=[]
        for i in range(N):
            s=sum(bsum[k][i] for k in sel);sq=sum(bsq[k][i] for k in sel);c=sum(bcnt[k] for k in sel)
            mu=s/c;v=sq/c-mu*mu;out.append(mu/math.sqrt(v) if v>1e-18 else 0.0)
        return out
    allset=set(range(S));lamneg=0;tot=0;ploss=0;xs=[];ys=[]
    for IS in combinations(range(S),S//2):
        OO=list(allset-set(IS));si=sr(list(IS));so=sr(OO)
        j=max(range(N),key=lambda i:si[i])
        rank=sorted(range(N),key=lambda i:so[i]).index(j)+1
        w=rank/(N+1.0);lam=math.log(w/(1-w))
        tot+=1
        if lam<0:lamneg+=1
        if so[j]<0:ploss+=1
        xs.append(si[j]);ys.append(so[j])
    mx=sum(xs)/len(xs);my=sum(ys)/len(ys)
    cov=sum((xs[k]-mx)*(ys[k]-my) for k in range(len(xs)));vx=sum((xs[k]-mx)**2 for k in range(len(xs)))
    return lamneg/tot, cov/vx, ploss/tot
# demos
demos={
 "overfit":   gen(12345,1000,80,[0.0]*80,0.01),
 "robust":    gen(999,1000,30,[0.0012]*3+[0.0]*27,0.01),
 "borderline":gen(555,1000,50,[0.0004]*6+[0.0]*44,0.01),
}
for name,M in demos.items():
    psr,annSR,SR0ann,j=dsr(M);pbo,slope,ploss=cscv(M,16)
    print(f"{name:11s} bestAnnSR={annSR:+.2f} DSR={psr:.3f} SR0ann={SR0ann:.2f} PBO={pbo:.2f} slope={slope:+.2f} P(OOS<0)={ploss:.2f}")
# algorithm cross-check sample value for JS comparison
print("CHECK overfit col_sr[0:3]:", [round(x,6) for x in col_sr(demos['overfit'])[:3]])
print("CHECK M_overfit[0][0:3]:", [round(demos['overfit'][0][i],6) for i in range(3)])
print("emax(80)=%.6f  norm_ppf(0.99)=%.6f  norm_cdf(1.088)=%.6f"%(emax(80),norm_ppf(0.99),norm_cdf(1.088)))
