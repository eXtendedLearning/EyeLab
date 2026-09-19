"""Re-run every load-bearing claim in the audit, independently, from a clean import."""
import numpy as np, cv2, sys, tempfile
from pathlib import Path
sys.path.insert(0,'.')
ok=lambda b: "PASS" if b else "*** FAIL ***"
R=[]

# S1a marker size
K=np.array([[900.,0,640],[0,900,360],[0,0,1]]); d5=np.zeros(5)
sq=lambda s:(lambda h:np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]],float))(s/2)
rv=np.array([[0.1],[0.2],[0.05]]); tv=np.array([[0.02],[0.01],[0.50]])
img,_=cv2.projectPoints(sq(0.020),rv,tv,K,d5)
_,rs,ts=cv2.solvePnP(sq(0.012),img.reshape(-1,2),K,d5,flags=cv2.SOLVEPNP_IPPE_SQUARE)
R.append(("S1a 12mm-vs-20mm gives -40% depth", abs(ts[2,0]/0.5-0.6)<0.01))

# S1b resolution
s=1280/1920; Kc=np.array([[1400.,0,960],[0,1400,540],[0,0,1]]); Kt=np.array([[1400*s,0,960*s],[0,1400*s,540*s],[0,0,1]])
from registration import marker_object_corners
O=np.vstack([marker_object_corners(c,[0,0,1],0.,0.012) for c in
   [np.array([0,0,0.]),np.array([.08,0,0.]),np.array([.08,.08,0.]),np.array([0,.08,0.])]]).astype(float)
i2,_=cv2.projectPoints(O,rv,np.array([[-.04],[-.04],[0.5]]),Kt,d5)
_,r2,t2=cv2.solvePnP(O,i2.reshape(-1,2),Kc,d5,flags=cv2.SOLVEPNP_ITERATIVE)
R.append(("S1b resolution mismatch -> >40% depth error", (t2[2,0]/0.5-1)>0.4))
import calibrate, inspect
R.append(("S1b load_calibration ignores image_width", "image_width" not in inspect.getsource(calibrate.load_calibration)))

# S1c units
import unv_to_json as U
src=inspect.getsource(U)
R.append(("S1c parser reads 'unit_code' (pyuff emits 'units_code')", "'unit_code'" in src or '"unit_code"' in src))
p=Path(tempfile.mkdtemp())/"mm.unv"; import pyuff
u=pyuff.UFF(str(p)); u.write_sets([pyuff.prepare_164(units_code=2,units_description="MM_N_S",temp_mode=1,length=0.001,force=1.0,temp=1.0,temp_offset=273.15),
 pyuff.prepare_2411(node_nums=np.array([1,2,3,4]),def_cs=np.zeros(4,int),disp_cs=np.zeros(4,int),color=np.zeros(4,int),
 x=np.array([0.,200.,200.,0.]),y=np.array([0.,0.,100.,100.]),z=np.zeros(4))],mode='overwrite')
D=U.UNVParser(p,validate_cs=False).parse()
R.append(("S1c mm file reports SI/1.0", D['units']['code']==1 and D['units']['lengthFactor']==1.0))
R.append(("S1c coords unscaled (200 not 0.2)", max(n['x'] for n in D['nodes'])==200.0))

# S2 coplanar rejection
from registration import SpatialRegistration, MarkerCorrespondence
tri=[[0,0,0],[0.1,0,0],[0,0.1,0]]
reg=SpatialRegistration([MarkerCorrespondence(marker_id=i,unv_position=np.array(p)) for i,p in enumerate(tri)])
Rt=np.array([[0,-1,0],[1,0,0],[0,0,1]],float); tt=np.array([0.5,0.2,1.0])
for i,pp in enumerate(tri): reg.update_detected_position(i,Rt@np.array(pp)+tt)
R.append(("S2 3 non-collinear coplanar -> compute() is None", reg.compute() is None))

# S3 gate bypass
from pose_estimator import PoseKalmanFilter, PoseResult
from pose_lock import PoseLock, PoseLockConfig
lk=PoseLock(PoseKalmanFilter(),PoseLockConfig())
g=PoseResult(rvec=np.array([[.05],[.1],[.02]]),tvec=np.array([[0.],[0.],[.5]]),marker_ids=[1,2,3],marker_count=3,rms_reproj_px=0.4)
for i in range(10): lk.process(g,now=i*.033)
b=PoseResult(rvec=np.array([[.05],[.80],[.02]]),tvec=np.array([[0.],[0.],[.80]]),marker_ids=[1,2,3],marker_count=3,rms_reproj_px=25.0)
o=lk.process(b,now=.363)
R.append(("S3 3-marker 25px/30cm-jump pose accepted", o is not None and not o.coasted))

# S4 quaternion
from pose_estimator import _rotation_matrix_to_quaternion as q
F=np.diag([1.,-1.,1.]); rng=np.random.default_rng(0); claim=alt=0
for _ in range(200):
    Rm=cv2.Rodrigues(rng.normal(0,1,3))[0]; Ru=F@Rm@np.linalg.inv(F)
    x,y,z,w=q(Rm); x2,y2,z2,w2=q(Ru); tgt=np.array([w2,x2,y2,z2])
    sm=lambda a,b: np.allclose(a,b,atol=1e-9) or np.allclose(a,-b,atol=1e-9)
    if sm(np.array([w,x,-y,-z]),tgt): claim+=1
    if sm(np.array([w,-x,y,-z]),tgt): alt+=1
R.append(("S4 THEORY quaternion rule wrong (0/200), (w,-x,y,-z) right (200/200)", claim==0 and alt==200))

# S5 ratio
rng=np.random.default_rng(7); pm=[];jn=[]
cs=[np.array([0,0,0.]),np.array([.08,0,0.]),np.array([.08,.08,0.]),np.array([0,.08,0.])]
rvt=np.array([[.05],[.12],[.02]]); tvt=np.array([[-.04],[-.04],[.5]])
for _ in range(200):
    AO=[];AI=[];ce=[]
    for c in cs:
        ow=marker_object_corners(c,[0,0,1],0.,0.012)
        im=cv2.projectPoints(ow,rvt,tvt,K,d5)[0].reshape(-1,2)+rng.normal(0,.2,(4,2))
        AO.append(ow);AI.append(im)
        _,_,t1=cv2.solvePnP(sq(0.012),im.reshape(-1,1,2).astype(np.float32),K,d5,flags=cv2.SOLVEPNP_IPPE_SQUARE); ce.append(t1.ravel())
    Rk,tk=SpatialRegistration._kabsch(np.array(cs),np.array(ce))
    Rtr=cv2.Rodrigues(rvt)[0]; truth=np.array([Rtr@x+tvt.ravel() for x in cs])
    pm.append(np.sqrt(np.mean(np.sum((truth-np.array([Rk@x+tk.ravel() for x in cs]))**2,1)))*1000)
    Oc=np.vstack(AO).astype(float);Ic=np.vstack(AI).astype(float)
    _,rb,tb=cv2.solvePnP(Oc,Ic,K,d5,flags=cv2.SOLVEPNP_ITERATIVE); rb,tb=cv2.solvePnPRefineLM(Oc,Ic,K,d5,rb,tb)
    Rb=cv2.Rodrigues(rb)[0]
    jn.append(np.sqrt(np.mean(np.sum((truth-np.array([Rb@x+tb.ravel() for x in cs]))**2,1)))*1000)
ratio=np.mean(pm)/np.mean(jn)
R.append((f"S5 per-marker Kabsch >5x worse than joint solve (measured {ratio:.1f}x)", ratio>5))

print("="*78); print("INDEPENDENT RE-VERIFICATION OF AUDIT CLAIMS"); print("="*78)
for n,v in R: print(f"  [{ok(v)}] {n}")
print("="*78); print(f"  {sum(1 for _,v in R if v)}/{len(R)} claims reproduced")
