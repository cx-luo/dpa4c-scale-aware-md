"""Regenerate six quantitative figures from the included CSV files only.

Python 3.12 / Matplotlib 3.11.2; no private ledger, network or device needed.
PDF is the manuscript format; PNG is a 300-dpi inspection copy.
"""
from pathlib import Path
import csv,sys
P=Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
 'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
BLUE='#0072B2';TEAL='#009E73';RED='#D55E00';GOLD='#E69F00';GRAY='#c4cbd1'
def rows(name):
    with (P/'data'/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(fig,name):
    (P/'figures').mkdir(exist_ok=True)
    fig.savefig(P/'figures'/f'{name}.pdf',bbox_inches='tight',pad_inches=.05,metadata={'CreationDate':None})
    fig.savefig(P/'figures'/f'{name}.png',dpi=300,bbox_inches='tight',pad_inches=.05)
    plt.close(fig)
n=rows('nine-results.csv');x=list(range(9))
fig,ax=plt.subplots(figsize=(7,2.5));ax.bar(x,[float(v['speedup']) for v in n],color=[BLUE]*3+[TEAL]*3+[GOLD]*3,width=.65)
ax.errorbar(x,[float(v['speedup']) for v in n],yerr=[[float(v['speedup'])-float(v['min_speedup']) for v in n],[float(v['max_speedup'])-float(v['speedup']) for v in n]],fmt='none',ecolor='black',capsize=3,lw=.8)
ax.set_xticks(x,[v['case'].replace('-','\n') for v in n]);ax.set_ylabel('Full-interval speedup');ax.set_ylim(0,6.7);ax.axhline(1,color='gray',ls=':',lw=.8)
for i,v in enumerate(n):ax.text(i,float(v['max_speedup'])+.15,f"{float(v['speedup']):.2f}",ha='center',fontsize=8)
fig.tight_layout();save(fig,'nine-speedup')

profiles=rows('profile-busy.csv');families_data=rows('profile-families.csv');families=[k for k in families_data[0] if k!='case']
fig,(ax,bx)=plt.subplots(1,2,figsize=(7,2.75),gridspec_kw={'width_ratios':[1,1.55]})
yy=list(range(len(profiles)));busy=[100*float(v['device_busy_fraction']) for v in profiles];idle=[100*float(v['device_idle_fraction']) for v in profiles]
ax.barh(yy,busy,color=BLUE,label='Device active');ax.barh(yy,idle,left=busy,color=GRAY,label='No active kernel');ax.set_yticks(yy,[v['case'] for v in profiles]);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Fraction of trace window (%)');ax.legend(loc='lower left',bbox_to_anchor=(-.05,1),frameon=False)
for i,v in enumerate(idle):ax.text(98,i,f'{v:.1f}',ha='right',va='center',fontsize=8)
mat=[[float(v[f]) for f in families] for v in families_data]
bx.imshow(mat,aspect='auto',cmap='Blues',vmin=0,vmax=50);bx.set_xticks(range(len(families)),[f.replace('_','\n') for f in families],fontsize=8);bx.set_yticks(yy,[]);bx.set_title('Family spans / window (%)\nNon-exclusive; overlap retained',fontsize=9)
for i,row in enumerate(mat):
    for j,v in enumerate(row):bx.text(j,i,f'{v:.1f}',ha='center',va='center',color='white' if v>27 else 'black',fontsize=8)
fig.tight_layout(w_pad=1);save(fig,'scale-profile')

d=rows('scope-effects.csv');xx=list(range(len(d)));fig,ax=plt.subplots(figsize=(7,2.6))
ax.bar([x-.18 for x in xx],[float(v['full_gain_pct']) for v in d],width=.35,color=BLUE,label='Full capture');ax.bar([x+.18 for x in xx],[float(v['md_gain_pct']) for v in d],width=.35,color=TEAL,hatch='//',label='Continuous MD')
ax.set_xticks(xx,[v['label'] for v in d]);ax.set_ylabel('Throughput improvement (%)');ax.axhline(0,color='black',lw=.7);ax.legend(frameon=False,ncol=2);fig.tight_layout();save(fig,'measurement-scope')

d=rows('capture-boundary.csv');vals=[float(v['seconds']) for v in d];fig,ax=plt.subplots(figsize=(3.35,2.3))
ax.barh([2,1,0],vals,color=[BLUE,GOLD,GRAY]);ax.set_yticks([2,1,0],['MD + output','SHA256 seal','Other']);ax.set_xlabel('Full-capture seconds')
for y,v in zip([2,1,0],vals):ax.text(v+.2,y,f'{v:.2f}',va='center',fontsize=8)
ax.set_xlim(0,26);fig.tight_layout();save(fig,'capture-boundary')

d=rows('historical-scores.csv');fig,ax=plt.subplots(figsize=(3.35,2.3));ax.plot(range(4),[float(v['score']) for v in d],marker='o',color=BLUE);ax.set_xticks(range(4),[v['label'] for v in d]);ax.set_ylabel('Measured development score');ax.set_ylim(3.51,3.81);ax.set_xlim(-.5,3.5)
for i,v in enumerate(d):ax.annotate(f"{float(v['score']):.4f}",(i,float(v['score'])),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
fig.tight_layout();save(fig,'historical-scores')

data=rows('negative-endpoints.csv');labels={'B01':'SourceOverlap*','B02':'Radix-lite*','B05-v1-Si-small':'Fit/force v1','B05-v2-Si-small':'Fit/force v2','B09':'ScratchBridge','B10':'Dense consumer','B06-65536-1':'Strassen register','B06-65536-2':'Strassen shared','B07-Si-large-v2':'Codec v2'}
fig,axs=plt.subplots(1,2,figsize=(7,3),gridspec_kw={'width_ratios':[1.45,1]})
for ax,title in zip(axs,['Continuous MD','Isolated layer / microstage']):
    subset=[r for r in data if r['endpoint']==title];vals=[float(r['throughput_change_pct']) for r in subset]
    ax.barh(range(len(subset)),vals,color=[RED if r['id'] in ['B01','B02'] else BLUE for r in subset]);ax.set_yticks(range(len(subset)),[labels[r['id']] for r in subset]);ax.invert_yaxis();ax.set_xlim(-65,1);ax.set_title(title,fontsize=9);ax.set_xlabel('Throughput change (%)');ax.axvline(0,color='black',lw=.7)
    for i,v in enumerate(vals):ax.text(v+1 if v<-15 else v-1,i,f'{v:.1f}',va='center',ha='left' if v<-15 else 'right',fontsize=8,color='white' if v<-15 else 'black')
fig.tight_layout(w_pad=1.5);save(fig,'endpoint-separated')
print('Rebuilt six data figures from included CSV files.')
