import numpy as np
import matplotlib.pyplot as plt
import math, random
from icecube import dataio, dataclasses, icetray


class Photons():
    def __init__(self, positions, ids, infos):
        self.x = positions[:,0]
        self.y = positions[:,1]
        self.z = positions[:,2]
        self.t = infos[:, 1]
        
        self.dx = infos[:, 2]
        self.dy = infos[:, 3]
        self.dz = infos[:, 4]
        
        self.string = ids[:,0]
        self.om     = ids[:,1]
        
        self.geo_string = ids[:,2]
        self.geo_om     = ids[:,3]
        
        self.mask = None
        self.check_sanity()
        
        self.pmt = -np.ones_like(self.x)
        
    def check_sanity(self):
        print "Checking sanity..."
        assert(sum(self.string == self.geo_string))
        assert(sum(self.om == self.geo_om))

        assert(len(self.x) == len(self.t))
        assert(len(self.x) == len(self.om))
        print "Passed everything"

    def set_mask(self, string, om=None, pmt=None):
        if om is None:
            self.mask = (self.string == string) 
        elif pmt is None:
            self.mask = (self.string == string)&(self.om == om)
        else:
            self.mask = (self.string == string)&(self.om == om)&(self.pmt == pmt)
        
    def del_mask(self):
        self.mask = None
        
    def get_pos(self, mode=0):
        x = self.x.copy()
        y = self.y.copy()
        z = self.z.copy()
        t = self.t.copy()
        if self.mask is not None:
            x = x[self.mask]
            y = y[self.mask]
            z = z[self.mask]
            t = t[self.mask]

        if mode == 0:
            return np.array([x, y, z, t]).T
        elif mode == 1:
            return x, y, z, t
        else:
            print "wrong mode %s " % mode
            print "set mode=0 or mode=1"

    def get_dir(self, mode=0):
        dx = self.dx.copy()
        dy = self.dy.copy()
        dz = self.dz.copy()
        if self.mask is not None:
            dx = dx[self.mask]
            dy = dy[self.mask]
            dz = dz[self.mask]

        if mode == 0:
            return np.array([dx, dy, dz]).T
        elif mode == 1:
            return dx, dy, dz
        else:
            print "wrong mode %s " % mode
            print "set mode=0 or mode=1"
            
    def get_ids(self, mode=0):
        string = self.string.copy()
        om = self.om.copy()
        pmt = self.pmt.copy()
        if self.mask is not None:
            string = string[self.mask]
            om = om[self.mask]
            pmt = pmt[self.mask]

        if mode == 0:
            return np.array([string, om, pmt]).T
        elif mode == 1:
            return string, om, pmt
        else:
            print "wrong mode %s " % mode
            print "set mode=0 or mode=1"            

        
class Geo():
    def __init__(self, pmts):
        self.string = pmts[:, 0]
        self.om = pmts[:, 1]
        self.pmt = pmts[:, 2]
        self.area = pmts[:, 3]
        self.radius = np.sqrt(self.area/np.pi)
        
        self.pmtx = pmts[:, 4]
        self.pmty = pmts[:, 5]
        self.pmtz = pmts[:, 6]
        
        self.pmtdx = pmts[:, 7]
        self.pmtdy = pmts[:, 8]
        self.pmtdz = pmts[:, 9]
        
        self.geox = pmts[:, 10]
        self.geoy = pmts[:, 11]
        self.geoz = pmts[:, 12]
        
        self.mask = None
        
    def set_mask(self, string, om, pmt=None):
        if om is None:
            self.mask = (self.string == string) 
        elif pmt is None:
            self.mask = (self.string == string)&(self.om == om)
        else:
            self.mask = (self.string == string)&(self.om == om)&(self.pmt == pmt)
        
    def del_mask(self):
        self.mask = None
        
    def get_pos_pmt(self, mode=0):
        x = self.pmtx.copy()
        y = self.pmty.copy()
        z = self.pmtz.copy()
        if self.mask is not None:
            x = x[self.mask]
            y = y[self.mask]
            z = z[self.mask]
        if mode == 0:
            return np.array([x, y, z]).T
        elif mode == 1:
            return x, y, z
        else:
            print "wrong mode %s " % mode
            print "set mode=0 or mode=1"
            
    def get_info_pmt(self, mode=0):
        A = self.area.copy()
        r = self.radius.copy()
        if self.mask is not None:
            A = A[self.mask]
            r = r[self.mask]

        if mode == 0:
            return np.array([A, r]).T
        elif mode == 1:
            return A, r
        else:
            print "wrong mode %s " % mode
            print "set mode=0 or mode=1"

    def get_dir_pmt(self, mode=0):
        dx = self.pmtdx.copy()
        dy = self.pmtdy.copy()
        dz = self.pmtdz.copy()
        if self.mask is not None:
            dx = dx[self.mask]
            dy = dy[self.mask]
            dz = dz[self.mask]

        if mode == 0:
            return np.array([dx, dy, dz]).T
        elif mode == 1:
            return dx, dy, dz
        else:
            print "wrong mode %s " % mode
            print "set mode=0 or mode=1"
            
    def get_pos_geo(self, mode=0):
        x = self.geox.copy()
        y = self.geoy.copy()
        z = self.geoz.copy()
        if self.mask is not None:
            geo_mask = self.mask & (self.pmt == 0)
            x = x[geo_mask][0]
            y = y[geo_mask][0]
            z = z[geo_mask][0]

        if mode == 0:
            return np.array([x, y, z]).T
        elif mode == 1:
            return x, y, z
        else:
            print "wrong mode %s " % mode
            print "set mode=0 or mode=1"
            
    def get_ids(self, mode=0):
        string = self.string.copy()
        om = self.om.copy()
        pmt = self.pmt.copy()

        if self.mask is not None:
            string = string[self.mask]
            om = om[self.mask]
            pmt = pmt[self.mask]


        if mode == 0:
            return np.array([string, om, pmt]).T
        elif mode == 1:
            return string, om, pmt, A
        else:
            print "wrong mode %s " % mode
            print "set mode=0 or mode=1" 


def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''
    Taken from: https://stackoverflow.com/a/50664367

    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

def I3OrientationToMatrix(ori):
    x = np.array([ori.up.x, ori.up.y, ori.up.z])
    y = np.array([ori.right.x, ori.right.y, ori.right.z])
    z = np.array([ori.dir.x, ori.dir.y, ori.dir.z])
    return np.array([-x, -y, z]).T

def I3DirToVec(I3Dir):
    return np.array([I3Dir.x, I3Dir.y, I3Dir.z])

try:
    class I3Geometry():
        
        def __init__(self, fn):
            from loading import load_I3Geometry
            omgeo, modulegeo = load_I3Geometry(fn)
            self.omgeo = omgeo
            self.modulegeo = modulegeo
            self.omkeys = omgeo.keys()
            self.omkeysArr = np.array(self.omkeys)
            self.Nkeys = len(self.omkeys)
            self.zShift = 0.01

        def get_geo(self, string=1, om=1, pmt=None):
            if pmt is None:
                mask = (self.omkeysArr[:,0] == string) & (self.omkeysArr[:,1] == om)
            else:
                mask = (self.omkeysArr[:,0] == string) & (self.omkeysArr[:,1] == om) & (self.omkeysArr[:,2] == pmt)

            omgeos = []
            inds = np.arange(len(self.omkeysArr))[mask]
            for ind in inds:
                omgeos.append(self.omgeo[self.omkeys[ind]])
            modulegeo = self.modulegeo[dataclasses.ModuleKey(string, om)]

            return omgeos, modulegeo


        def plot_pmt(self, ax, string=1, om=1, pmt=None, N=50, qlength=0.02):
            if pmt is None:
                mask = (self.omkeysArr[:,0] == string) & (self.omkeysArr[:,1] == om)
            else:
                mask = (self.omkeysArr[:,0] == string) & (self.omkeysArr[:,1] == om) & (self.omkeysArr[:,2] == pmt)

            inds = np.arange(len(self.omkeysArr))[mask]
            for ind in inds:
                key = self.omkeys[ind]
                pmt = self.omgeo[key]
                ori    = pmt.orientation
                A = I3OrientationToMatrix(ori)
                r = np.sqrt(pmt.area/np.pi)
                pmtDir = A[:,2]
                pmtCenter = np.array(pmt.position)
                pmtCenter[2] += self.zShift*pmtDir[2]/abs(pmtDir[2])

                Cpmt = make_circle(center=pmtCenter, r=r, rotation=A, N=N)
                q = np.hstack([pmtCenter, pmtDir])

                ax.plot(*Cpmt.T, c="g")
                ax.quiver(*q, length=qlength)

            set_axes_equal(ax)


        def get_pmt_circles(self, string=1, om=1, pmt=None, N=50, qlength=0.02, zShift=None):
            if pmt is None:
                mask = (self.omkeysArr[:,0] == string) & (self.omkeysArr[:,1] == om)
            else:
                mask = (self.omkeysArr[:,0] == string) & (self.omkeysArr[:,1] == om) & (self.omkeysArr[:,2] == pmt)

            if zShift is None:
                zShift = self.zShift

            circles = []
            inds = np.arange(len(self.omkeysArr))[mask]
            for ind in inds:
                key = self.omkeys[ind]
                pmt = self.omgeo[key]
                ori    = pmt.orientation
                A = I3OrientationToMatrix(ori)
                r = np.sqrt(pmt.area/np.pi)
                pmtDir = A[:,2]
                pmtCenter = np.array(pmt.position)
                pmtCenter[2] += zShift*pmtDir[2]/abs(pmtDir[2])

                Cpmt = make_circle(center=pmtCenter, r=r, rotation=A, N=N)
                circles.append(Cpmt)
            return circles
except:
    pass

def plot_dir_scan(dirScan, Geo, probName, cmap="plasma"):
    from numpy.linalg import norm

    omgeos, modulegeo = Geo.get_geo(93, 5, 0)
    geoCenter = modulegeo.pos
    geoR      = modulegeo.radius
    pmtCenter = omgeos[0].position
    pmtR = np.sqrt(omgeos[0].area/np.pi)

    Arot = I3OrientationToMatrix(omgeos[0].orientation)
    Cpmt = make_circle(center=pmtCenter, r=pmtR, rotation=Arot)

    pmtHits = dirScan["pmtHit"].values
    mask = pmtHits != 0
    pPos = dirScan[["posx", "posy", "posz"]].values.copy() - pmtCenter
    pDir = dirScan[["dirx", "diry", "dirz"]].values.copy()
    prob = dirScan[probName].values.copy()
    prob[mask] = 0

    nx = Arot[:,0]
    anglesPos = np.arccos(nx.dot(pPos.T)/norm(pPos, axis=1))*180/np.pi - 90
    anglesDir = np.arccos(nx.dot(pDir.T)/norm(pDir, axis=1))*180/np.pi - 90


    vmin = 0#prob[prob>1e-8].min() #- 0.05
    vmax = prob.max()
    vmin, vmax

    nx, ny = 200, 200
    anglesPos = np.reshape(anglesPos, (nx, ny))
    anglesDir = np.reshape(anglesDir, (nx, ny))
    prob = np.reshape(prob, (nx, ny))

    plt.figure()
    plt.pcolormesh(anglesPos, anglesDir, prob, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()

    # plt.title("Detection Probability")
    plt.title(probName)
    plt.xlabel(r"$\theta_{pos}$", size=15)
    plt.ylabel(r"$\theta_{dir}$", size=15)



def colormap_to_plotly(cm_name):
    import matplotlib
    from matplotlib import cm
    import numpy as np

    plasma_cmap = matplotlib.cm.get_cmap(cm_name)

    plasma_rgb = []
    norm = matplotlib.colors.Normalize(vmin=0, vmax=255)
    for i in range(0, 255):
        k = matplotlib.colors.colorConverter.to_rgb(plasma_cmap(norm(i)))
        plasma_rgb.append(k)

    def matplotlib_to_plotly(cmap, pl_entries):
        h = 1.0/(pl_entries-1)
        pl_colorscale = []

        for k in range(pl_entries):
            C = map(np.uint8, np.array(cmap(k*h)[:3])*255)
            pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

        return pl_colorscale

    plasma = matplotlib_to_plotly(plasma_cmap, 255)
    
    return plasma


def calc_angle(v1, v2, res="deg"):
    if res == "deg":
        return np.arccos(v1.dot(v2)/(norm(v1)*norm(v2)))*180/np.pi
    elif res == "rad":
        return np.arccos(v1.dot(v2)/(norm(v1)*norm(v2)))


def calc_thetas(pmtCenter, geoCenter, geoR, A):
    b = norm(pmtCenter - geoCenter)
    a = np.sqrt(geoR**2 - b**2)

    Ax = A[:,0]
    p1 = pmtCenter + a*Ax
    p2 = pmtCenter - a*Ax

    v1 = geoCenter - p1
    v2 = geoCenter - p2

    tmin = calc_angle(Ax, v1, res="rad")
    tmax =  calc_angle(Ax, v2, res="rad")

    return tmin, tmax


def plot_basis(A, ax, center=np.zeros(3), length=0.1):
    import matplotlib as mpl
    v = mpl.__version__

    colors = ["r", "g", "b"]
    if int(v.split(".")[0]) >= 2:
        for i in range(3):
            v = A[:,i]
            q = np.hstack([center, v])
            ax.quiver(*q.T, color=colors[i], length=length)
    else:
        for i in range(3):
            v = A[:,i]
            q = np.hstack([center+v*length, v])
            ax.quiver(*q.T, color=colors[i], length=length)

def make_circle(center=np.zeros(3), r=1.0, N=50, tmin=0, tmax=2*np.pi, rotation=None):
    t = np.linspace(tmin, tmax, N)
    x = r*np.cos(t)
    y = r*np.sin(t)
    z = t*0
    c = np.array([x, y, z])

    if not rotation is None:
        c = rotation.dot(c)

    c = c.T  + center
    return c

def swap_axes(A, swap="xyz"):
    assert len(swap) == 3
    s2int = {"x":0, "y":1, "z":2}
    s1 = s2int[swap[0]]
    s2 = s2int[swap[1]]
    s3 = s2int[swap[2]]
    Aswap = np.zeros_like(A)

    Aswap[:,0] = A[:, s1]
    Aswap[:,1] = A[:, s2]
    Aswap[:,2] = A[:, s3]

    return Aswap


def plot_circle(C, ax, p=np.zeros(3), v=None, color="b", onlyv=False, vl=0.025, scatter=False):
    C = C.copy().T + p
    if not onlyv:
        if scatter:
            ax.scatter(C[:,0], C[:,1], C[:,2], color=color)
        else:
            ax.plot(C[:,0], C[:,1], C[:,2], color=color, lw=3, zorder=3)
    if v is not None:
        ax.quiver(*list(p) + list(v), color="b", length=vl, normalize=True)

        
def plot_circle_surface(C, ax, p=np.zeros(3), v=None, color="b", onlyv=False):
    C = C.copy().T
    C[:,:,0] += p
    C[:,:,1] += p
    if not onlyv:
        ax.plot_surface(C[:,0,:], C[:,1,:], C[:,2,:], shade=False, edgecolor=color, color=color)
    if v is not None:
        ax.quiver(*list(p) + list(v), color="b", length=0.025, normalize=True)


def gram_schmidt(v1):
    v2 = np.array([1, 0, 0], dtype=float)
    v3 = np.array([0, 1, 0], dtype=float)
    A = np.array([v1, v2, v3]).T

    def normalize(v):
        return v / np.sqrt(v.dot(v))

    n = len(A)

    A[:, 0] = normalize(A[:, 0])

    for i in range(1, n):
        Ai = A[:, i]
        for j in range(0, i):
            Aj = A[:, j]
            t = Ai.dot(Aj)
            Ai = Ai - t * Aj
        A[:, i] = normalize(Ai)

    A = np.array([A[:,1], A[:,2], A[:,0]]).T
    
    return A


def transform_circle_3d(v1, r, N):
    t = np.linspace(0, 2*np.pi, N)
    x = r*np.cos(t)
    y = r*np.sin(t)
    z = 0*t

    C = np.array([x, y, z])
    A = gram_schmidt(v1.copy())
    Ct = A.dot(C)
    
    return Ct


def circle_surface(v1, r=1.0, N=100):
    return np.array([transform_circle_3d(v1, r, N) for r in [r, 0.0001]])


def find_pmt(p_pos, p_dir, geos):

    glassThickness = 0.001
    moduleElongation = 0.001
    
    omRadiusOrigin = 0.1651
    omRadius = omRadiusOrigin + moduleElongation
    
    assert (omRadius < 1.0)
    
    omRadiusSquared = omRadius*omRadius

    geo_center = geos.get_pos_geo()

    pmt_poss = geos.get_pos_pmt()
    pmt_dirs = geos.get_dir_pmt()
    pmt_As = geos.get_info_pmt()[:,0]

    px= p_pos[0] - geo_center[0]
    py= p_pos[1] - geo_center[1]
    pz= p_pos[2] - geo_center[2]
    pr2 = px*px + py*py + pz*pz;

    dx = p_dir[0]
    dy = p_dir[1]
    dz = p_dir[2]


    px_starting = px
    py_starting = py
    pz_starting = pz

    distFromDOMCenter = np.sqrt(pr2)

    #     print distFromDOMCenter, distFromDOMCenter - omRadius

    pr_scale = omRadius/distFromDOMCenter

    #     print pr_scale

    px *= pr_scale
    py *= pr_scale
    pz *= pr_scale
    pr2 = omRadiusSquared

    oversizeQECorrection = pr_scale*pr_scale

    #     print oversizeQECorrection

    dot = px*dx + py*dy + pz*dz
    #     print dot

    if dot > 0:
        return -1

    
    shift_x = px - px_starting
    shift_y = py - py_starting
    shift_z = pz - pz_starting
    timingCorrection = (shift_x*dx + shift_y*dy + shift_z*dz)/2.29e8   
    
    
    pathLengthInOM = float("nan")
    pathLengthInGlass = float("nan")
    distFromPMTCenterSquaredFound = float("nan")
    pmtRadiusSquaredFound = float("nan")


    foundIntersection=-1;
    for i, (pmt_pos, pmt_dir, A) in enumerate(zip(pmt_poss, pmt_dirs, pmt_As)):
        
        pmtArea = A
        pmtRadiusSquared = pmtArea/np.pi
        
        nx = pmt_dir[0]
        ny = pmt_dir[1]
        nz = pmt_dir[2]

        nl = nx*nx + ny*ny + nz*nz

        # find the intersection of the PMT's surface plane and the photon's path
        denom = dx*nx + dy*ny + dz*nz


        if denom>=1e-8:
            continue

        
        ax = pmt_pos[0] - geo_center[0]
        ay = pmt_pos[1] - geo_center[1]
        az = (pmt_pos[2] + moduleElongation*nz/abs(nz)) - geo_center[2]
        
        mu = ((ax-px)*nx + (ay-py)*ny + (az-pz)*nz)/denom

        if mu < 0.:
            continue

        
        distFromPMTCenterSquared = (ax-px-mu*dx)*(ax-px-mu*dx) + \
                                   (ay-py-mu*dy)*(ay-py-mu*dy) + \
                                   (az-pz-mu*dz)*(az-pz-mu*dz)          


        # Make sure that it hit the plane within the radius of the disc
#         print distFromPMTCenterSquared, pmtRadiusSquared
        if distFromPMTCenterSquared > pmtRadiusSquared:
            continue

        # See if we found another intersection
        if foundIntersection > 0.:
            if np.isnan(pathLengthInOM) or mu < pathLengthInOM:
                pass
            else:
                continue


        sx = px + mu*dx
        sy = py + mu*dy
        sz = pz + mu*dz - moduleElongation*nz/abs(nz)

        s_dot_d = sx*dx + sy*dy + sz*dz
        s_dot_s = sx*sx + sy*sy + sz*sz
        
        pathLengthInOM = s_dot_d + np.sqrt(s_dot_d*s_dot_d + np.power(omRadiusOrigin, 2) - s_dot_s)
        pathLengthThroughGel = s_dot_d + np.sqrt(s_dot_d*s_dot_d + np.power(omRadiusOrigin - glassThickness, 2) - s_dot_s)
        pathLengthInGlass = pathLengthInOM - pathLengthThroughGel

        foundIntersection = i

    return foundIntersection





def fibonacci_sphere(samples=1,randomize=True):
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x,y,z])

    return np.array(points)


def random_on_sphere(N=1):
    points = []
    for i in range(N):
        dir_theta = np.random.rand()*2.0*np.pi
        vz = 2.0*np.random.rand()-1.0
        vx = np.sqrt(1.0-pow(vz,2))*np.cos(dir_theta)
        vy = np.sqrt(1.0-pow(vz,2))*np.sin(dir_theta)
        points.append([vx, vy, vz])
    return np.array(points)



