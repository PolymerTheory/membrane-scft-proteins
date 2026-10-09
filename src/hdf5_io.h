#pragma once

#ifdef USE_HDF5

#include <hdf5.h>

// HDF5 / XDMF helpers for restart and visualization output.
// Restart fields (`wins.h5`) are written in either float32 or float64,
// while visualization fields are written in float32. The current
// implementation gathers data to rank 0 and writes atomically via a
// temporary file followed by rename().

static bool hdf5_parallel_build_available()
{
#ifdef H5_HAVE_PARALLEL
    return true;
#else
    return false;
#endif
}

static bool hdf5_dataset_exists(hid_t file_id, const char *name)
{
    return H5Lexists(file_id, name, H5P_DEFAULT) > 0;
}

static hid_t create_hdf5_file(const char *fname)
{
    return H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
}

static bool write_dataset4d(hid_t file_id, const char *name, hid_t dtype, const void *data,
                            const hsize_t d0, const hsize_t d1, const hsize_t d2, const hsize_t d3,
                            const int deflate_level)
{
    hsize_t dims[4] = {d0, d1, d2, d3};
    hsize_t chunk[4] = {1, d1, d2, d3};
    hid_t space_id = H5Screate_simple(4, dims, NULL);
    if(space_id < 0) return false;

    hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    if(dcpl_id < 0){
        H5Sclose(space_id);
        return false;
    }
    H5Pset_chunk(dcpl_id, 4, chunk);
    H5Pset_shuffle(dcpl_id);
    H5Pset_deflate(dcpl_id, deflate_level);

    hid_t dset_id = H5Dcreate2(file_id, name, dtype, space_id, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
    if(dset_id < 0){
        H5Pclose(dcpl_id);
        H5Sclose(space_id);
        return false;
    }
    herr_t status = H5Dwrite(dset_id, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(dset_id);
    H5Pclose(dcpl_id);
    H5Sclose(space_id);
    return status >= 0;
}

static void write_atomic_text_file(const char *fname, const std::string &text)
{
    std::string tmp = std::string(fname) + ".tmp";
    FILE *fp = fopen(tmp.c_str(), "w");
    if(fp==NULL) return;
    fputs(text.c_str(), fp);
    fclose(fp);
    rename(tmp.c_str(), fname);
}

static std::string xdmf_hyperslab_for_dataset(const char *h5name, const char *dset_name,
                                              const int ii, const int m0, const int m1, const int m2,
                                              const char *number_type, const int precision)
{
    char buf[4096];
    snprintf(
        buf, sizeof(buf),
        "<DataItem ItemType=\"HyperSlab\" Dimensions=\"%d %d %d\" Type=\"HyperSlab\">\n"
        "  <DataItem Dimensions=\"3 4\" Format=\"XML\">\n"
        "    %d 0 0 0\n"
        "    1 1 1 1\n"
        "    1 %d %d %d\n"
        "  </DataItem>\n"
        "  <DataItem Dimensions=\"%d %d %d %d\" NumberType=\"%s\" Precision=\"%d\" Format=\"HDF\">%s:/%s</DataItem>\n"
        "</DataItem>\n",
        m0, m1, m2,
        ii, m0, m1, m2,
        strn, m0, m1, m2, number_type, precision, h5name, dset_name
    );
    return std::string(buf);
}

static void write_wins_xdmf(const char *fname, const char *h5name, const int *m, const double *D, const int precision_bits)
{
    const char *number_type = "Float";
    const int precision = (precision_bits == 32) ? 4 : 8;
    double dx = D[0]/m[0];
    double dy = D[1]/m[1];
    double dz = D[2]/m[2];
    std::string text;
    text += "<?xml version=\"1.0\" ?>\n";
    text += "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    text += "<Xdmf Version=\"3.0\">\n";
    text += " <Domain>\n";
    text += "  <Grid Name=\"Wins\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";
    for(int ii=0; ii<strn; ii++){
        char header[512];
        snprintf(header, sizeof(header),
                 "   <Grid Name=\"replica_%d\" GridType=\"Uniform\">\n"
                 "    <Time Value=\"%d\"/>\n"
                 "    <Topology TopologyType=\"3DCORECTMesh\" Dimensions=\"%d %d %d\"/>\n"
                 "    <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n"
                 "     <DataItem Dimensions=\"3\" Format=\"XML\">0 0 0</DataItem>\n"
                 "     <DataItem Dimensions=\"3\" Format=\"XML\">%.10g %.10g %.10g</DataItem>\n"
                 "    </Geometry>\n",
                 ii, ii, m[0], m[1], m[2], dx, dy, dz);
        text += header;
        text += "    <Attribute Name=\"W1\" AttributeType=\"Scalar\" Center=\"Node\">\n";
        text += xdmf_hyperslab_for_dataset(h5name, "W1", ii, m[0], m[1], m[2], number_type, precision);
        text += "    </Attribute>\n";
        text += "    <Attribute Name=\"W2\" AttributeType=\"Scalar\" Center=\"Node\">\n";
        text += xdmf_hyperslab_for_dataset(h5name, "W2", ii, m[0], m[1], m[2], number_type, precision);
        text += "    </Attribute>\n";
        text += "    <Attribute Name=\"Wp\" AttributeType=\"Scalar\" Center=\"Node\">\n";
        text += xdmf_hyperslab_for_dataset(h5name, "Wp", ii, m[0], m[1], m[2], number_type, precision);
        text += "    </Attribute>\n";
        text += "   </Grid>\n";
    }
    text += "  </Grid>\n";
    text += " </Domain>\n";
    text += "</Xdmf>\n";
    write_atomic_text_file(fname, text);
}

static void write_concentrations_xdmf(const char *fname, const char *h5name, const int *m, const double *D)
{
    const char *number_type = "Float";
    const int precision = 4;
    double dx = D[0]/m[0];
    double dy = D[1]/m[1];
    double dz = D[2]/m[2];
    std::string text;
    text += "<?xml version=\"1.0\" ?>\n";
    text += "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    text += "<Xdmf Version=\"3.0\">\n";
    text += " <Domain>\n";
    text += "  <Grid Name=\"Concentrations\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";
    for(int ii=0; ii<strn; ii++){
        char header[512];
        snprintf(header, sizeof(header),
                 "   <Grid Name=\"replica_%d\" GridType=\"Uniform\">\n"
                 "    <Time Value=\"%d\"/>\n"
                 "    <Topology TopologyType=\"3DCORECTMesh\" Dimensions=\"%d %d %d\"/>\n"
                 "    <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n"
                 "     <DataItem Dimensions=\"3\" Format=\"XML\">0 0 0</DataItem>\n"
                 "     <DataItem Dimensions=\"3\" Format=\"XML\">%.10g %.10g %.10g</DataItem>\n"
                 "    </Geometry>\n",
                 ii, ii, m[0], m[1], m[2], dx, dy, dz);
        text += header;
        const char *names[] = {"rhoA", "rhoB", "rhoH"};
        for(int jj=0; jj<3; jj++){
            text += "    <Attribute Name=\"";
            text += names[jj];
            text += "\" AttributeType=\"Scalar\" Center=\"Node\">\n";
            text += xdmf_hyperslab_for_dataset(h5name, names[jj], ii, m[0], m[1], m[2], number_type, precision);
            text += "    </Attribute>\n";
        }
        text += "   </Grid>\n";
    }
    text += "  </Grid>\n";
    text += " </Domain>\n";
    text += "</Xdmf>\n";
    write_atomic_text_file(fname, text);
}

static void write_concentrations_derived_xdmf(const char *fname, const char *h5name, const int *m, const double *D)
{
    const char *number_type = "Float";
    const int precision = 4;
    double dx = D[0]/m[0];
    double dy = D[1]/m[1];
    double dz = D[2]/m[2];
    std::string text;
    text += "<?xml version=\"1.0\" ?>\n";
    text += "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    text += "<Xdmf Version=\"3.0\">\n";
    text += " <Domain>\n";
    text += "  <Grid Name=\"ConcentrationsDerived\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";
    for(int ii=0; ii<strn; ii++){
        char header[512];
        snprintf(header, sizeof(header),
                 "   <Grid Name=\"replica_%d\" GridType=\"Uniform\">\n"
                 "    <Time Value=\"%d\"/>\n"
                 "    <Topology TopologyType=\"3DCORECTMesh\" Dimensions=\"%d %d %d\"/>\n"
                 "    <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n"
                 "     <DataItem Dimensions=\"3\" Format=\"XML\">0 0 0</DataItem>\n"
                 "     <DataItem Dimensions=\"3\" Format=\"XML\">%.10g %.10g %.10g</DataItem>\n"
                 "    </Geometry>\n",
                 ii, ii, m[0], m[1], m[2], dx, dy, dz);
        text += header;
        const char *names[] = {"rhoA", "rhoB", "rhoH"};
        for(int jj=0; jj<3; jj++){
            text += "    <Attribute Name=\"";
            text += names[jj];
            text += "\" AttributeType=\"Scalar\" Center=\"Node\">\n";
            text += xdmf_hyperslab_for_dataset(h5name, names[jj], ii, m[0], m[1], m[2], number_type, precision);
            text += "    </Attribute>\n";
        }
        char dims[64];
        snprintf(dims, sizeof(dims), "%d %d %d", m[0], m[1], m[2]);
        text += "    <Attribute Name=\"rhoAmB\" AttributeType=\"Scalar\" Center=\"Node\">\n";
        text += "     <DataItem ItemType=\"Function\" Function=\"$0 - $1\" Dimensions=\"";
        text += dims;
        text += "\">\n";
        text += xdmf_hyperslab_for_dataset(h5name, "rhoA", ii, m[0], m[1], m[2], number_type, precision);
        text += xdmf_hyperslab_for_dataset(h5name, "rhoB", ii, m[0], m[1], m[2], number_type, precision);
        text += "     </DataItem>\n";
        text += "    </Attribute>\n";
        text += "   </Grid>\n";
    }
    text += "  </Grid>\n";
    text += " </Domain>\n";
    text += "</Xdmf>\n";
    write_atomic_text_file(fname, text);
}

static void write_proteins_xdmf(const char *fname, const char *h5name, const int *m, const double *D)
{
    const char *number_type = "Float";
    const int precision = 4;
    double dx = D[0]/m[0];
    double dy = D[1]/m[1];
    double dz = D[2]/m[2];
    std::string text;
    text += "<?xml version=\"1.0\" ?>\n";
    text += "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    text += "<Xdmf Version=\"3.0\">\n";
    text += " <Domain>\n";
    text += "  <Grid Name=\"Proteins\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";
    for(int ii=0; ii<strn; ii++){
        char header[512];
        snprintf(header, sizeof(header),
                 "   <Grid Name=\"replica_%d\" GridType=\"Uniform\">\n"
                 "    <Time Value=\"%d\"/>\n"
                 "    <Topology TopologyType=\"3DCORECTMesh\" Dimensions=\"%d %d %d\"/>\n"
                 "    <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n"
                 "     <DataItem Dimensions=\"3\" Format=\"XML\">0 0 0</DataItem>\n"
                 "     <DataItem Dimensions=\"3\" Format=\"XML\">%.10g %.10g %.10g</DataItem>\n"
                 "    </Geometry>\n",
                 ii, ii, m[0], m[1], m[2], dx, dy, dz);
        text += header;
        const char *names[] = {"prot1", "prot2", "prot3"};
        for(int jj=0; jj<3; jj++){
            text += "    <Attribute Name=\"";
            text += names[jj];
            text += "\" AttributeType=\"Scalar\" Center=\"Node\">\n";
            text += xdmf_hyperslab_for_dataset(h5name, names[jj], ii, m[0], m[1], m[2], number_type, precision);
            text += "    </Attribute>\n";
        }
        text += "   </Grid>\n";
    }
    text += "  </Grid>\n";
    text += " </Domain>\n";
    text += "</Xdmf>\n";
    write_atomic_text_file(fname, text);
}

static bool write_wins_hdf5_atomic(const char *fname, double **W, const int *m, const int checkpoint_precision_bits)
{
    std::string tmp = std::string(fname) + ".tmp";
    hid_t file_id = create_hdf5_file(tmp.c_str());
    if(file_id < 0) return false;
    const size_t total = static_cast<size_t>(strn) * m[0] * m[1] * m[2];
    bool ok = true;
    if(checkpoint_precision_bits == 32){
        std::vector<float> w1(total), w2(total), wp(total);
        for(int ii=0; ii<strn; ii++)
            for(int r=0; r<M; r++){
                const size_t idx = static_cast<size_t>(ii) * M + r;
                w1[idx] = static_cast<float>(W[ii][r]);
                w2[idx] = static_cast<float>(W[ii][r+M]);
                wp[idx] = static_cast<float>(W[ii][r+2*M]);
            }
        ok = write_dataset4d(file_id, "W1", H5T_NATIVE_FLOAT, w1.data(), strn, m[0], m[1], m[2], 9);
        ok = ok && write_dataset4d(file_id, "W2", H5T_NATIVE_FLOAT, w2.data(), strn, m[0], m[1], m[2], 9);
        ok = ok && write_dataset4d(file_id, "Wp", H5T_NATIVE_FLOAT, wp.data(), strn, m[0], m[1], m[2], 9);
    } else {
        std::vector<double> w1(total), w2(total), wp(total);
        for(int ii=0; ii<strn; ii++)
            for(int r=0; r<M; r++){
                const size_t idx = static_cast<size_t>(ii) * M + r;
                w1[idx] = W[ii][r];
                w2[idx] = W[ii][r+M];
                wp[idx] = W[ii][r+2*M];
            }
        ok = write_dataset4d(file_id, "W1", H5T_NATIVE_DOUBLE, w1.data(), strn, m[0], m[1], m[2], 9);
        ok = ok && write_dataset4d(file_id, "W2", H5T_NATIVE_DOUBLE, w2.data(), strn, m[0], m[1], m[2], 9);
        ok = ok && write_dataset4d(file_id, "Wp", H5T_NATIVE_DOUBLE, wp.data(), strn, m[0], m[1], m[2], 9);
    }
    H5Fclose(file_id);
    if(!ok) return false;
    rename(tmp.c_str(), fname);
    return true;
}

static bool write_concentrations_hdf5_atomic(const char *fname, const int *m,
                                             double **phiA, double **phiB, double **phih)
{
    std::string tmp = std::string(fname) + ".tmp";
    hid_t file_id = create_hdf5_file(tmp.c_str());
    if(file_id < 0) return false;
    const size_t total = static_cast<size_t>(strn) * m[0] * m[1] * m[2];
    std::vector<float> rhoA(total), rhoB(total), rhoH(total);
    for(int ii=0; ii<strn; ii++)
        for(int r=0; r<M; r++){
            const size_t idx = static_cast<size_t>(ii) * M + r;
            rhoA[idx] = static_cast<float>(phiA[ii][r]);
            rhoB[idx] = static_cast<float>(phiB[ii][r]);
            rhoH[idx] = static_cast<float>(phih[ii][r]);
        }
    bool ok = true;
    ok = ok && write_dataset4d(file_id, "rhoA", H5T_NATIVE_FLOAT, rhoA.data(), strn, m[0], m[1], m[2], 9);
    ok = ok && write_dataset4d(file_id, "rhoB", H5T_NATIVE_FLOAT, rhoB.data(), strn, m[0], m[1], m[2], 9);
    ok = ok && write_dataset4d(file_id, "rhoH", H5T_NATIVE_FLOAT, rhoH.data(), strn, m[0], m[1], m[2], 9);
    H5Fclose(file_id);
    if(!ok) return false;
    rename(tmp.c_str(), fname);
    return true;
}

static bool write_proteins_hdf5_atomic(const char *fname, const int *m,
                                       double **prott1, double **prott2, double **prott3)
{
    std::string tmp = std::string(fname) + ".tmp";
    hid_t file_id = create_hdf5_file(tmp.c_str());
    if(file_id < 0) return false;
    const size_t total = static_cast<size_t>(strn) * m[0] * m[1] * m[2];
    std::vector<float> p1(total), p2(total), p3(total);
    for(int ii=0; ii<strn; ii++)
        for(int r=0; r<M; r++){
            const size_t idx = static_cast<size_t>(ii) * M + r;
            p1[idx] = static_cast<float>(prott1[ii][r]);
            p2[idx] = static_cast<float>(prott2[ii][r]);
            p3[idx] = static_cast<float>(prott3[ii][r]);
        }
    bool ok = true;
    ok = ok && write_dataset4d(file_id, "prot1", H5T_NATIVE_FLOAT, p1.data(), strn, m[0], m[1], m[2], 9);
    ok = ok && write_dataset4d(file_id, "prot2", H5T_NATIVE_FLOAT, p2.data(), strn, m[0], m[1], m[2], 9);
    ok = ok && write_dataset4d(file_id, "prot3", H5T_NATIVE_FLOAT, p3.data(), strn, m[0], m[1], m[2], 9);
    H5Fclose(file_id);
    if(!ok) return false;
    rename(tmp.c_str(), fname);
    return true;
}

static bool read_dataset4d_to_double(hid_t file_id, const char *name, std::vector<double> &out,
                                     int &d0, int &d1, int &d2, int &d3)
{
    hid_t dset_id = H5Dopen2(file_id, name, H5P_DEFAULT);
    if(dset_id < 0) return false;
    hid_t space_id = H5Dget_space(dset_id);
    hsize_t dims[4];
    int ndims = H5Sget_simple_extent_dims(space_id, dims, NULL);
    if(ndims != 4){
        H5Sclose(space_id);
        H5Dclose(dset_id);
        return false;
    }
    d0 = static_cast<int>(dims[0]);
    d1 = static_cast<int>(dims[1]);
    d2 = static_cast<int>(dims[2]);
    d3 = static_cast<int>(dims[3]);
    hid_t dtype = H5Dget_type(dset_id);
    H5T_class_t cls = H5Tget_class(dtype);
    size_t sz = H5Tget_size(dtype);
    const size_t total = static_cast<size_t>(d0) * d1 * d2 * d3;
    out.resize(total);
    herr_t status = -1;
    if(cls == H5T_FLOAT && sz == sizeof(float)){
        std::vector<float> tmp(total);
        status = H5Dread(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp.data());
        if(status >= 0) for(size_t i=0; i<total; i++) out[i] = tmp[i];
    } else {
        status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data());
    }
    H5Tclose(dtype);
    H5Sclose(space_id);
    H5Dclose(dset_id);
    return status >= 0;
}

static int read_wins_hdf5(const char *fname, double **W, const int *m)
{
    hid_t file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    if(file_id < 0) return -1;
    bool hasW1 = hdf5_dataset_exists(file_id, "W1");
    bool hasW2 = hdf5_dataset_exists(file_id, "W2");
    bool hasWp = hdf5_dataset_exists(file_id, "Wp");
    if(!hasW1 || !hasWp){
        H5Fclose(file_id);
        return -1;
    }
    int d0=0, d1=0, d2=0, d3=0;
    std::vector<double> w1, w2, wp;
    if(!read_dataset4d_to_double(file_id, "W1", w1, d0, d1, d2, d3)){
        H5Fclose(file_id);
        return -1;
    }
    if(d0 != strn || d1 != m[0] || d2 != m[1] || d3 != m[2]){
        H5Fclose(file_id);
        return -1;
    }
    if(!read_dataset4d_to_double(file_id, "Wp", wp, d0, d1, d2, d3)){
        H5Fclose(file_id);
        return -1;
    }
    int nfields = 2;
    if(hasW2){
        if(!read_dataset4d_to_double(file_id, "W2", w2, d0, d1, d2, d3)){
            H5Fclose(file_id);
            return -1;
        }
        nfields = 3;
    }
    H5Fclose(file_id);
    for(int ii=0; ii<strn; ii++)
        for(int r=0; r<M; r++){
            const size_t idx = static_cast<size_t>(ii) * M + r;
            W[ii][r] = w1[idx];
            W[ii][r+M] = (nfields == 3) ? w2[idx] : w1[idx];
            W[ii][r+2*M] = wp[idx];
        }
    return nfields;
}

#else

static bool hdf5_parallel_build_available() { return false; }
static bool write_wins_hdf5_atomic(const char *, double **, const int *, const int) { return false; }
static bool write_concentrations_hdf5_atomic(const char *, const int *, double **, double **, double **) { return false; }
static bool write_proteins_hdf5_atomic(const char *, const int *, double **, double **, double **) { return false; }
static void write_wins_xdmf(const char *, const char *, const int *, const double *, const int) {}
static void write_concentrations_xdmf(const char *, const char *, const int *, const double *) {}
static void write_concentrations_derived_xdmf(const char *, const char *, const int *, const double *) {}
static void write_proteins_xdmf(const char *, const char *, const int *, const double *) {}
static int read_wins_hdf5(const char *, double **, const int *) { return -2; }

#endif
