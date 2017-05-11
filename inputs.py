import cPickle as pickle
import h5py
import numpy as np
import healpy as hp

def partial2full(partial, obspix, nside, fill_with_nan=True):
    """
    Convert partial map into full sky map

    Parameters
    ----------
        * partial: 1D array, the observed data
        * obspix: 1D array, the label of observed pixels
        * nside: int, nside of the partial data
        * fill_with_nan: boolean, if True it fills with nan unobserved
            pixels (it allows compression when saving on the disk)

    Output
    ----------
        * full: 1D array, full sky map

    """
    if fill_with_nan is True:
        full = np.zeros(12 * nside**2) * np.nan
    else:
        full = np.zeros(12 * nside**2)
    full[obspix] = partial
    return full

class HealpixMapInfo(object):
    def __init__(self, npix, obspix, nside, source):
        assert npix == len(obspix), 'Wrong number of pixels!'
        self.npix = npix
        self.obspix = obspix
        self.source = source
        self.nside = nside

class InputScan():
    def __init__(self, mapinfo):
        self.mapinfo = HealpixMapInfo(
            mapinfo.npix,
            mapinfo.obspix,
            mapinfo.nside,
            mapinfo.source)

        npix = self.mapinfo.npix

        # To accumulate A^T N^-1 A
        self.w = np.zeros(npix)
        self.cc = np.zeros(npix)
        self.cs = np.zeros(npix)
        self.ss = np.zeros(npix)

        self.nhit = np.zeros(npix, dtype=np.int32)

    def save(self, fn):
        """
        Yo!
        """
        f = h5py.File(fn, 'w')
        npix = (self.mapinfo.npix, )
        t = '=f8'
        c = 'gzip'
        d = f.create_dataset('w', npix, t, compression=c)
        d[:] = self.w
        d = f.create_dataset('cc', npix, t, compression=c)
        d[:] = self.cc
        d = f.create_dataset('cs', npix, t, compression=c)
        d[:] = self.cs
        d = f.create_dataset('ss', npix, t, compression=c)
        d[:] = self.ss
        d = f.create_dataset('nhit', npix, '=i4', compression=c)
        d[:] = self.nhit

        mapinfos = pickle.dumps(self.mapinfo)
        f.attrs['mapinfo'] = mapinfos

        f.close()

    @staticmethod
    def load(fn):
        """
        Yo!
        """
        f = h5py.File(fn, 'r')
        mapinfos = f.attrs['mapinfo']
        mapinfo = pickle.loads(mapinfos)
        mapvec = InputScan(mapinfo)
        mapvec.w = f['w'][:]
        mapvec.cc = f['cc'][:]
        mapvec.cs = f['cs'][:]
        mapvec.ss = f['ss'][:]
        mapvec.nhit = f['nhit'][:]
        f.close()
        return mapvec

    @staticmethod
    def change_resolution(m_in, nside_out):
        print 'Changing the resolution of the map from %d to %d' % \
            (m_in.mapinfo.nside, nside_out)
        tmp = hp.ud_grade(
            partial2full(
                m_in.nhit,
                m_in.mapinfo.obspix,
                m_in.mapinfo.nside), nside_out)
        obspix = np.where(tmp > 0)[0]

        def replace(field, obspix_in, obspix_out, nside_in, nside_out):
            tmp = hp.ud_grade(
                partial2full(field, obspix_in, nside_in), nside_out)
            return tmp[obspix_out]

        m_in.nhit = replace(
            m_in.nhit,
            m_in.mapinfo.obspix,
            obspix,
            m_in.mapinfo.nside,
            nside_out)
        m_in.w = replace(
            m_in.w,
            m_in.mapinfo.obspix,
            obspix,
            m_in.mapinfo.nside,
            nside_out)
        m_in.cc = replace(
            m_in.cc,
            m_in.mapinfo.obspix,
            obspix,
            m_in.mapinfo.nside,
            nside_out)
        m_in.cs = replace(
            m_in.cs,
            m_in.mapinfo.obspix,
            obspix,
            m_in.mapinfo.nside,
            nside_out)
        m_in.ss = replace(
            m_in.ss,
            m_in.mapinfo.obspix,
            obspix,
            m_in.mapinfo.nside,
            nside_out)

        m_in.mapinfo.nside = nside_out
        m_in.mapinfo.obspix = obspix
        m_in.mapinfo.npix = len(obspix)

        return m_in


class InputScan_full():
    def __init__(self, mapinfo):
        self.mapinfo = mapinfo

        npix = self.mapinfo.npix
        self.npix = npix

        # To accumulate A^T N^-1 d
        self.d = np.zeros(npix)
        self.dc = np.zeros(npix)
        self.ds = np.zeros(npix)

        # To accumulate A^T N^-1 A
        self.w = np.zeros(npix)
        self.cc = np.zeros(npix)
        self.cs = np.zeros(npix)
        self.ss = np.zeros(npix)

        self.nhit = np.zeros(npix, dtype=np.int32)

    def save(self, fn):
        f = h5py.File(fn, 'w')
        npix = (self.npix,)
        t = '=f8'
        c = 'gzip'
        d = f.create_dataset('I', npix, t, compression=c)
        d[:] = self.I
        d = f.create_dataset('Q', npix, t, compression=c)
        d[:] = self.Q
        d = f.create_dataset('U', npix, t, compression=c)
        d[:] = self.U
        d = f.create_dataset('w', npix, t, compression=c)
        d[:] = self.w
        d = f.create_dataset('cc', npix, t, compression=c)
        d[:] = self.cc
        d = f.create_dataset('cs', npix, t, compression=c)
        d[:] = self.cs
        d = f.create_dataset('ss', npix, t, compression=c)
        d[:] = self.ss
        d = f.create_dataset('d', npix, t, compression=c)
        d[:] = self.d
        d = f.create_dataset('dc', npix, t, compression=c)
        d[:] = self.dc
        d = f.create_dataset('ds', npix, t, compression=c)
        d[:] = self.ds
        d = f.create_dataset('nhit', npix, '=i4', compression=c)
        d[:] = self.nhit

        mapinfos = pickle.dumps(self.mapinfo)
        f.attrs['mapinfo'] = mapinfos

        f.close()

    @staticmethod
    def load(fn):
        f = h5py.File(fn, 'r')
        mapinfos = f.attrs['mapinfo']
        mapinfo = pickle.loads(mapinfos)
        mapvec = InputScan_full(mapinfo)
        mapvec.I = f['I'][:]
        mapvec.Q = f['Q'][:]
        mapvec.U = f['U'][:]
        mapvec.w = f['w'][:]
        mapvec.cc = f['cc'][:]
        mapvec.cs = f['cs'][:]
        mapvec.ss = f['ss'][:]
        mapvec.d = f['d'][:]
        mapvec.dc = f['dc'][:]
        mapvec.ds = f['ds'][:]
        mapvec.nhit = f['nhit'][:]
        f.close()
        return mapvec
