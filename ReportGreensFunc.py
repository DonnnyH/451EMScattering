"""
Author: Harry Donegan
Solution of classical electromagnetic scattering problem for the interaction of laguerre-Gaussian beams interacting with a collection of point dipoles
Follows the formalism set out in:
'Cooperative resonances in light scattering from two-dimensional atomic arrays'
https://doi.org/10.1103/PhysRevLett.118.113601 (Supplemental information)
arxiv link: https://arxiv.org/abs/1610.00138
"""
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import time
import cmasher as cmr


def GreensFunc(r1,r2, delta):
    """
    Electromagnetic dyadic Greens function in matrix form
    r1 - position vector of point 1
    r2 - position vector of point 2
    """
    k=2*np.pi
    r = np.linalg.norm(r1 - r2)
    if r==0:
        return 0j*np.eye(3)
    else:
        rel = (r1-r2)/r
        return -3/(2*(2*delta+1j))*np.exp(1j*k*r)/(k*r)*((np.eye(3)-np.outer(rel,rel))+(1j/(k*r)-1/(k*r)**2)*(np.eye(3)-3*np.outer(rel,rel)))
    
def polarisation(i, sigma):
    """
    Returns unit polarisaion vector of the E-field
    sigma = +(-)1 - Right(left)-handed circular polarisation 
    sigma = 0 - Linear polarisation in x-direction
    """
    if sigma==0:
        if (i)%3==0:
            return 1
        else:
         return 0
    elif sigma == 1:
        if (i)%3==0:
            return 1/np.sqrt(2)
        elif (i)%3==1:
            return +1j/np.sqrt(2)
        else: 
            return 0
    elif sigma == -1:
        if (i)%3==0:
            return 1/np.sqrt(2)
        elif (i)%3==1:
            return -1j/np.sqrt(2)
        else: 
            return 0

def LG_intensity(r,z,theta, l, p, w0):
    """
    Electric field pattern of a Laguerre-Gaussian laser beam (without polarisation)
    r - radius from beam axis
    z - distance along beam axis
    l - azimuthal index
    p - radial index
    w0 - beam waist
    """
    z_r=np.pi*w0**2
    def w(z):
        return w0*np.sqrt(1+(z/z_r)**2)
    def R(z):
        if z==0:
            return np.inf
        else:
            return z*(1+(z_r/z)**2)
    def G(z):
        return (np.abs(l)+2*p+1)*np.arctan(z/z_r)
    phase = np.exp(1j*(l*theta - G(z) + 2*np.pi*r**2/(2*R(z))+2*np.pi*z))
    E=np.sqrt(2*np.math.factorial(p)/(np.pi*np.math.factorial(p+np.abs(l))))*(w0/w(z))*(np.sqrt(2)*r/w(z))**(np.abs(l))*sp.eval_genlaguerre(p,np.abs(l), (2*r**2/w(z)**2))*np.exp(-r**2/(w(z)**2))*phase
    return E

def square_array(N, d):
    """
    Returns the positions of atoms in the XY plane in an NxN square
    d - interatomic distance in units of wavelengths
    """
    R=[]
    a=d/2
    for i in range(N):
        for j in range(N):
            R.append(np.array([-(N*a)+2*a*i+a,-(N*a)+2*a*j+a, 0]))
    return R

def ScatteringMatrix(R, l, p, w0, sigma):
    """
    Calculates the self-consistent solutions of the dipole moments for each atom from the scattering equation
    """
    n = len(R)
    G = np.zeros(shape=(3*n,3*n), dtype=complex)
    E = np.zeros(shape=(3*n,1), dtype=complex)
    for i in range(3*n):
        x = R[i//3][0]
        y = R[i//3][1]
        z = R[i//3][2]
        r = np.sqrt(x**2+y**2)
        theta = np.arctan2(y,x)
        E[i]=(LG_intensity(r,z,theta,l,p,w0))*polarisation(i,sigma)
    for i in range(3*n):
        for j in range(3*n):
            G[i][j]=GreensFunc(R[(i)//3], R[(j)//3], 0)[i%3][j%3]
    M = np.eye(3*n, dtype=complex)-G
    if n==0:
        return np.array([[0],[0],[0]])
    else:
        In = np.linalg.inv(M)
        E1 = In@E
        return np.array_split(E1,n)

def LG_intensity_Scat(E, R,r,z,theta, l, p, w0, x, y, sigma):
    """
    Computes the field after the scattering solution has been solved by including the induced fields of the dipoles
    """
    E0 = LG_intensity(r,z,theta, l, p, w0)*np.array([[1],[1],[1]], dtype=complex)
    for i in range(3):
        E0[i][0] = E0[i][0]*polarisation(i,sigma)
    q = np.array([x,y,z])
    if len(R)==0:
        E0=E0
    else:
        for i in range(len(R)):
            E0 = E0+GreensFunc(R[i],q, 0)@ E[i]
    def reshaper(M):
        E = [0+0*0j, 0+0*0j, 0+0*0j]
        for i in range(len(M)):
            E[i]=M[i][0]
        return E
    return reshaper(E0)

def Reflected_Beam(E, R,r,z,theta, l, p, w0, x, y):
    """
    Computes the field after the scattering solution has been solved by including the induced fields of the dipoles
    """
    E0 = np.array([[0],[0],[0]], dtype=complex)
    q = np.array([x,y,z])
    if len(R)==0:
        E0=E0
    else:
        for i in range(len(R)):
            E0 = E0+GreensFunc(R[i],q, 0)@ E[i]
    def reshaper(M):
        E = [0+0*0j, 0+0*0j, 0+0*0j]
        for i in range(len(M)):
            E[i]=M[i][0]
        return E
    return reshaper(E0)

def reflectivity(R, acc, l, p, w0, sigma):
    """
    Computes the reflectivity of the array under excitation by taking the quotient of the integral of the scattered field for z>0 over the solid angle of the hemisphere to the integral of the incident field for z<0 over the solid angle of the hemisphere
    To compute the integral over the spheres, we calculate the field at discrete points over the hemisphere and apply the Lebedev quadrature
    int(f(\omega), d\omega)=2pi/N*sum(f(\omega_i)),
    where N is the number of grid points
    See https://en.wikipedia.org/wiki/Lebedev_quadrature
    """
    def normvect(thet, phi):
        """
        Unit Vector of normal to sphere
        thet - azimuthal angle [0,2*pi)
        phi - polar angle [0, pi)
        """
        return np.array([np.sin(thet)*np.cos(phi),np.sin(thet)*np.sin(phi),np.cos(thet)])
    Escat = ScatteringMatrix(R, l, p, w0, sigma)
    
    phi=0
    rad = 100
    refl = []
    for i in range(acc):
        thet = 0.01
        for i in range(acc):
            x = rad*np.sin(thet)*np.cos(phi)
            y = rad*np.sin(thet)*np.sin(phi)
            z= rad*np.cos(thet)
            r = np.sqrt(x**2+y**2)
            theta = np.arctan2(y,x)
            refl.append(np.dot(normvect(thet, phi),LG_intensity_Scat(Escat, R, r,z, theta, l, p, w0, x, y)))
            thet = thet + (np.pi-0.01)/(2*acc)
        phi = phi + 2*np.pi/acc
    ref = np.abs(np.sum(refl))**2   
        
    phi=0
    inc = []
    for i in range(acc):
        thet = 0.01
        for i in range(acc):
            x = rad*np.sin(thet)*np.cos(phi)
            y= rad*np.sin(thet)*np.sin(phi)
            z= rad*np.cos(thet)
            r = np.sqrt(x**2+y**2)
            theta = np.arctan2(y,x)
            inc.append(np.dot(normvect(thet, phi),np.array(LG_intensity(r, z, theta, l, p, w0))))
            thet = thet + (np.pi-0.01)/(2*acc)
        phi = phi + 2*np.pi/acc
    incident =np.abs(np.sum(inc))**2
    return ref, incident

def normalisation(l,p,w0):
    """
    Provided that the beam is a 'doughnut' LG beam (p=0) then this function normalises the beams intensity at its maximum (r=sqrt(l/2)*w0, z=0)
    """
    z_r=np.pi*w0**2
    def w(z):
        return w0*np.sqrt(1+(z/z_r)**2)
    def R(z):
        if z==0:
            return np.inf
        else:
            return z*(1+(z_r/z)**2)
    def G(z):
        return (np.abs(l)+2*p+1)*np.arctan(z/z_r)
    if p==0:
        E0=np.sqrt(2*np.math.factorial(p)/(np.pi*np.math.factorial(p+np.abs(l))))*(w0/w(0))*(np.sqrt(2)*(np.sqrt(np.abs(l))*w0/np.sqrt(2))/w(0))**(np.abs(l))*sp.eval_genlaguerre(p,np.abs(l), (2*(np.sqrt(np.abs(l))*w0/np.sqrt(2))**2/w(0)**2))*np.exp(-(np.sqrt(np.abs(l))*w0/np.sqrt(2))**2/(w(0)**2))
    else:
        E0=1
    return E0

N=0
a=0.5
R=square_array(N, a)
l=1
p=0
sigma = 0

w0=0.3*a*N
w0=0.27
start = time.time()
Escat = ScatteringMatrix(R, l, p, w0, sigma)

m=150

pos1 = np.zeros(shape=(m,m), dtype=complex)

x=-4
y=0
for i in range(m):
    z = -4
    for j in range(m):
        r = np.sqrt(x**2+y**2)
        theta = np.arctan2(y,x)
        I=(np.linalg.norm(LG_intensity_Scat(Escat, R, r,z, theta, l, p, w0, x, 0, sigma))**2)/np.abs(normalisation(l, p, w0))**2
        if I>0.1:
            I=0.1
        pos1[i][j]=I
        z = z+ 8/m
    x = x + 8/m


x1list = np.linspace(-4,4,m)
z1list = np.linspace(-4,4,m)
X1, Z1 = np.meshgrid(z1list, x1list)



N=0
a=0.5
R=square_array(N, a)
l=0
p=0
sigma = 0

w0=0.3*a*N
w0=0.27
start = time.time()
Escat = ScatteringMatrix(R, l, p, w0, sigma)

m=150

pos = np.zeros(shape=(m,m), dtype=complex)

x=-4
y=0
for i in range(m):
    z = -4
    for j in range(m):
        r = np.sqrt(x**2+y**2)
        theta = np.arctan2(y,x)
        I=(np.linalg.norm(LG_intensity_Scat(Escat, R, r,z, theta, l, p, w0, x, 0, sigma))**2)/np.abs(normalisation(l, p, w0))**2
        if I>0.1:
            I=0.1
        pos[i][j]=I
        z = z+ 8/m
    x = x + 8/m


xlist = np.linspace(-4,4,m)
zlist = np.linspace(-4,4,m)
X, Z = np.meshgrid(zlist, xlist)


from matplotlib.colors import LinearSegmentedColormap
my_colormap = LinearSegmentedColormap.from_list('my colormap', [(0,'#ffffff'),(0.15,'#bcb6ff'),(1,'#5829a7')], 1000)
cmap = cmr.amethyst         # CMasher
cmap1 = plt.get_cmap('cmr.amethyst')
cmap1_reversed = cmap1.reversed()

fig, ax = plt.subplots(1,1)
cp = ax.contourf(X, Z, pos, levels =150, cmap=cmap1_reversed)



"""
fig, ax = plt.subplots(1,2)
c1=ax[0].contourf(X1, Z1, pos1, levels=150, cmap=cmap1_reversed)
c2=ax[1].contourf(X, Z, pos, levels=150, cmap=cmap1_reversed)
for ax in fig.get_axes():
    ax.set(xlabel=r'$z/\lambda$', ylabel=r'$x/\lambda$')
    ax.label_outer()
"""
"""
cbar = fig.colorbar(ax[0], ticks=[0,0.1])
cbar.set_label(r'$\frac{|\mathbf{E}|^2}{|\mathbf{E}_{0}|^2}$', rotation=0, size=15)
"""
plt.show()
end = time.time()
print(end - start)

