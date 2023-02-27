from nutils import cli, mesh, function, export
import numpy as np
import matplotlib.collections, matplotlib.colors

#-----------------
# Main starts here
#-----------------
def main( L        = 2  ,  # Length of domain
          H        = 2  ,  # Height of domain
          nelems   = 1  ,  # Number of elements
          npatches = 4  ,  # Number of patches
          degree   = 2  ,  # Polynomial degree of basis function
          ):

    # Multipatch information
    patches, patchverts, cps, w = multipatch_info(L, H)

    # Construct mesh
    domain, geom = mesh.multipatch(patches=patches, patchverts=patchverts, nelems=nelems)

    # Construct B-spline basis function
    basis = domain.basis('spline', degree=degree, patchcontinuous=False)

    # Weight function
    weightfunc = basis.dot(w)

    # Geometry
    nurbsbasis = basis * w / weightfunc
    geom = (nurbsbasis[:,np.newaxis] * cps).sum(0)

    # Jump check
    jumpval = abs(domain.interfaces['interpatch'].sample('bezier', 5).eval(function.jump(geom))).max()
    assert jumpval == 0.0, 'Topology jump check failed'

    # Patch ids
    patchIDs = domain.basis('patch').dot(np.arange(npatches))

    # Sample points
    bezier = domain.sample('bezier', 4)
    points, patchID = bezier.eval([geom, patchIDs])

    # Create a colormap and match the patchIDs TODO
    cdict = {'blue': '#0077BB', 'green': '#009988', 'orange': '#EE7733', 'red': '#CC3311'}

    with export.mplfigure('NURBS-multipatch.png', dpi=300) as fig:
        ax  = fig.add_subplot(111, title='Patches')
        im  = ax.tripcolor(points[:,0], points[:,1], bezier.tri, patchID, cmap='Paired')
        ax.add_collection(matplotlib.collections.LineCollection(points[bezier.hull,:2], colors='k', linewidths=1, alpha=1 if bezier.tri is None else .5))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_aspect('equal')

def patch_cps(Lmin, Lmax, Hmin, Hmax):
    cps = np.array([[Lmin,Hmin], [Lmin,Hmax/2], [Lmin, Hmax],      \
                    [Lmax/2,Hmin], [Lmax/2,Hmax/2], [Lmax/2,Hmax], \
                    [Lmax,Hmin], [Lmax,Hmax/2], [Lmax,Hmax]])
    return cps

def multipatch_info(L, H):
    patchverts = np.array([[0, 0], [0,H/2], [0,H/2], [0,H], \
                           [L/2,0], [L/2,H/2], [L/2,H/2], [L/2,H], \
                           [L,0], [L,H/2], [L,H] ])


    patches = np.array([[0,1,4,5], \
                        [4,5,8,9], \
                        [6,7,9,10], \
                        [2,3,6,7]  ])

    cps_0 = patch_cps(0, L/2, 0, H/2)
    cps_1 = patch_cps(L/2, L, 0, H/2)
    cps_2 = patch_cps(L/2, L, H/2, H)
    cps_3 = patch_cps(0, L/2, H/2, H)

    cps = np.concatenate((cps_0,cps_1,cps_2,cps_3), axis=0)
    w   = np.ones(cps.shape[0])*np.sqrt(2)

    return patches, patchverts, cps, w

cli.run(main)