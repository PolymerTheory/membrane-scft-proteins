#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <errno.h>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cctype>
#include <stdint.h>
#include <string.h>

using namespace std;

#ifndef SCFT_STRN
#define SCFT_STRN 48
#endif
const int strn=SCFT_STRN;

int m[3], M;
double D[3], pi=4*atan(1.0);
double Pr=0.0, Py=0.0, Pth=0.0;
double *Px0, *Py0, *Pz0, *Pth0;
double *protein_Pth_replica=NULL;
double protein_Prx=0.24;
double pro1=2.0, pro2=0.0, pro3=0.4, protein_mvin=0.48;
double protein_pitch=0.0;
double protein_patch_wx=0.209;
double protein_patch_wy=0.253;
double protein_patch_offset2=0.0;
double protein_patch_offset3=0.0;
int protein_enabled=1, protein_movable=1;
int protein_hydrophilic_mode=0;
int protein_symmetrize=0;
int protein_pivot_align=0;
int protein_cap_mode=1;
int snare_n=8;
double snare_x0=0.0;
double snare_ring_radius=-1.0;
double snare_length=0.50;
double snare_radius=0.30;
double snare_pad=0.50;
double **snare_positions=NULL;
int snare_positions_loaded=0;
int protein_have_Pth_file=0;
double **prott1, **prott2, **prott3;

typedef std::unordered_map<std::string, std::vector<double>> ParamMap;

std::string trim_copy(const std::string& s){
    size_t start = 0;
    while(start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) start++;
    size_t end = s.size();
    while(end > start && std::isspace(static_cast<unsigned char>(s[end-1]))) end--;
    return s.substr(start, end-start);
}

bool parse_numeric_tokens(const std::string& text, std::vector<double>& values){
    std::istringstream iss(text);
    std::string tok;
    while(iss >> tok){
        char *end = NULL;
        errno = 0;
        double value = strtod(tok.c_str(), &end);
        if(end==tok.c_str() || *end!='\0' || errno==ERANGE) return false;
        values.push_back(value);
    }
    return true;
}

bool fexist (const char *filename){
    if (FILE * file = fopen(filename, "r")){
        fclose(file);
        return true;
    }
    return false;
}

void fail_read(const char *fname, const char *detail){
    fprintf(stderr, "Fatal read error in '%s': %s\n", fname, detail);
    exit(1);
}

void fail_read(int procid, const char *fname, const char *detail){
    (void)procid;
    fail_read(fname, detail);
}

int as_int_checked(double value, const char *fname, const char *key){
    double rounded = floor(value + 0.5);
    if(fabs(value-rounded) > 1E-9){
        char detail[256];
        sprintf(detail, "key '%s' must contain an integer value", key);
        fail_read(fname, detail);
    }
    return static_cast<int>(rounded);
}

ParamMap readParameters(const char *fname){
    ParamMap params;
    std::ifstream file(fname);
    if(!file.is_open()) fail_read(fname, "failed to open keyed input file");
    std::string line;
    int lineno = 0;
    while(std::getline(file, line)){
        lineno++;
        size_t commentPos = line.find('#');
        if(commentPos != std::string::npos) line.erase(commentPos);
        line = trim_copy(line);
        if(line.empty()) continue;
        size_t eqPos = line.find('=');
        if(eqPos == std::string::npos){
            char detail[256];
            sprintf(detail, "keyed input line %d is missing '='", lineno);
            fail_read(fname, detail);
        }
        std::string key = trim_copy(line.substr(0, eqPos));
        std::string rhs = trim_copy(line.substr(eqPos+1));
        std::vector<double> values;
        if(!parse_numeric_tokens(rhs, values)){
            char detail[256];
            sprintf(detail, "keyed input line %d contains a non-numeric value", lineno);
            fail_read(fname, detail);
        }
        params[key] = values;
    }
    return params;
}

bool get_optional_param(const ParamMap& params, const char *key, size_t expected_count, std::vector<double>& values){
    ParamMap::const_iterator it = params.find(key);
    if(it == params.end()) return false;
    if(it->second.size() != expected_count) return false;
    values = it->second;
    return true;
}

bool get_optional_int_param(const ParamMap& params, const char *key, int &value, const char *fname){
    ParamMap::const_iterator it = params.find(key);
    if(it == params.end()) return false;
    if(it->second.size() != 1){
        char detail[256];
        sprintf(detail, "key '%s' expected 1 value but found %lu", key, (unsigned long)it->second.size());
        fail_read(fname, detail);
    }
    value = as_int_checked(it->second[0], fname, key);
    return true;
}

bool looks_like_keyed_input(const char *fname){
    std::ifstream file(fname);
    std::string line;
    while(std::getline(file, line)){
        size_t commentPos = line.find('#');
        if(commentPos != std::string::npos) line.erase(commentPos);
        line = trim_copy(line);
        if(line.empty()) continue;
        return line.find('=') != std::string::npos;
    }
    return false;
}

void parse_keyed_input(const char *fname){
    ParamMap params = readParameters(fname);
    const std::vector<double>& chi_vals = params["chi"];
    (void)chi_vals;
    const std::vector<double>& m_vals = params["m"];
    const std::vector<double>& D_vals = params["D"];
    if(m_vals.size()!=3 || D_vals.size()!=3) fail_read(fname, "keyed input requires m and D with 3 values each");
    m[0] = as_int_checked(m_vals[0], fname, "m");
    m[1] = as_int_checked(m_vals[1], fname, "m");
    m[2] = as_int_checked(m_vals[2], fname, "m");
    D[0] = D_vals[0];
    D[1] = D_vals[1];
    D[2] = D_vals[2];
}

void parse_legacy_input(const char *fname){
    std::ifstream file(fname);
    if(!file.is_open()) fail_read(fname, "failed to open legacy input file");
    std::string line;
    std::vector< std::vector<double> > rows;
    while(std::getline(file, line)){
        size_t commentPos = line.find('#');
        if(commentPos != std::string::npos) line.erase(commentPos);
        line = trim_copy(line);
        if(line.empty()) continue;
        std::vector<double> values;
        if(!parse_numeric_tokens(line, values)) fail_read(fname, "legacy input contains a non-numeric value");
        rows.push_back(values);
    }
    if(rows.size()<3) fail_read(fname, "legacy input must contain at least 3 numeric lines");
    if(rows[1].size()!=4 || rows[2].size()!=3) fail_read(fname, "legacy input lines 2 and 3 must contain N/m and D");
    m[0] = as_int_checked(rows[1][1], fname, "m");
    m[1] = as_int_checked(rows[1][2], fname, "m");
    m[2] = as_int_checked(rows[1][3], fname, "m");
    D[0] = rows[2][0];
    D[1] = rows[2][1];
    D[2] = rows[2][2];
}

void parse_keyed_protein_input(const char *fname){
    ParamMap params = readParameters(fname);
    std::vector<double> values;
    if(get_optional_param(params, "P", 3, values)){
        Pr = values[0];
        Py = values[1];
        Pth = values[2];
    }
    if(get_optional_param(params, "Prx", 1, values)) protein_Prx = values[0];
    if(get_optional_param(params, "P0", 4, values)){
        for(int ii=0; ii<strn; ii++){
            Px0[ii] = values[0];
            Py0[ii] = values[1];
            Pz0[ii] = values[2];
            Pth0[ii] = values[3];
        }
    }
    if(get_optional_param(params, "pro", 3, values)){
        pro1 = values[0];
        pro2 = values[1];
        pro3 = values[2];
    }
    if(get_optional_param(params, "mvin", 1, values)) protein_mvin = values[0];
    if(get_optional_param(params, "pitch", 1, values)) protein_pitch = values[0];
    if(get_optional_param(params, "patch_wx", 1, values)) protein_patch_wx = values[0];
    if(get_optional_param(params, "patch_wy", 1, values)) protein_patch_wy = values[0];
    if(get_optional_param(params, "patch_offset2", 1, values)) protein_patch_offset2 = values[0];
    if(get_optional_param(params, "patch_offset3", 1, values)) protein_patch_offset3 = values[0];
    get_optional_int_param(params, "protein_enabled", protein_enabled, fname);
    get_optional_int_param(params, "protein_movable", protein_movable, fname);
    get_optional_int_param(params, "hydrophilic_mode", protein_hydrophilic_mode, fname);
    get_optional_int_param(params, "protein_symmetrize", protein_symmetrize, fname);
    get_optional_int_param(params, "protein_pivot_align", protein_pivot_align, fname);
    get_optional_int_param(params, "protein_cap_mode", protein_cap_mode, fname);
    get_optional_int_param(params, "snare_n", snare_n, fname);
    if(get_optional_param(params, "snare_x0", 1, values)) snare_x0 = values[0];
    if(get_optional_param(params, "snare_ring_radius", 1, values)) snare_ring_radius = values[0];
    if(get_optional_param(params, "snare_length", 1, values)) snare_length = values[0];
    if(get_optional_param(params, "snare_radius", 1, values)) snare_radius = values[0];
    if(get_optional_param(params, "snare_pad", 1, values)) snare_pad = values[0];
}

void validate_protein_settings(const char *source_name)
{
    if(!(protein_enabled==0 || protein_enabled==1))
        fail_read(source_name, "protein_enabled must be 0 or 1");
    if(!(protein_movable==0 || protein_movable==1))
        fail_read(source_name, "protein_movable must be 0 or 1");
    if(protein_Prx<=0.0)
        fail_read(source_name, "Prx must be positive");
    if(!(protein_hydrophilic_mode==0 || protein_hydrophilic_mode==1))
        fail_read(source_name, "hydrophilic_mode must be 0 (surface) or 1 (patch)");
    if(!(protein_symmetrize>=0 && protein_symmetrize<=2))
        fail_read(source_name, "protein_symmetrize must be 0 (single), 1 (add), or 2 (legacy copy)");
    if(!(protein_cap_mode==0 || protein_cap_mode==1))
        fail_read(source_name, "protein_cap_mode must be 0 (none) or 1 (aligned)");
    if(!(protein_pivot_align==0 || protein_pivot_align==1))
        fail_read(source_name, "protein_pivot_align must be 0 or 1");
    if(protein_patch_wx<=0.0)
        fail_read(source_name, "patch_wx must be positive");
    if(protein_patch_wy<=0.0)
        fail_read(source_name, "patch_wy must be positive");
    if(snare_n<=0)
        fail_read(source_name, "snare_n must be positive");
    if(snare_length<=0.0)
        fail_read(source_name, "snare_length must be positive");
    if(snare_radius<=0.0)
        fail_read(source_name, "snare_radius must be positive");
    if(snare_pad<0.0)
        fail_read(source_name, "snare_pad must be non-negative");
}

void malloc2d(double ***ARAY, int l1, int l2){
    *ARAY = (double **)malloc(l1 * sizeof(double *));
    for(int i=0;i<l1;i++) (*ARAY)[i] = (double *)malloc(l2 * sizeof(double));
}

void tovtk (char *fname, int *m, double *D, double *data)
{
    int x,y,z,r;
    FILE *out;
    out = fopen(fname,"w");
    fprintf(out,"# vtk DataFile Version 2.0\n");
    fprintf(out,"CT Density\n");
    fprintf(out,"ASCII\n\n");
    fprintf(out,"DATASET STRUCTURED_POINTS\n");
    fprintf(out,"DIMENSIONS %d %d %d\n",m[2],m[1],m[0]);
    fprintf(out,"ORIGIN 0.000000 0.000000 0.000000\n");
    fprintf(out,"SPACING %lf %lf %lf\n\n",D[2]/m[2],D[1]/m[1],D[0]/m[0]);
    fprintf(out,"POINT_DATA %ld\n",(long) m[0]*m[1]*m[2]);
    fprintf(out,"SCALARS scalars float\n");
    fprintf(out,"LOOKUP_TABLE default\n\n");
    for (x=0; x<m[0]; x++)
        for (y=0; y<m[1]; y++){
            for (z=0; z<m[2]; z++) {
                r = z+ y*m[2] + x*m[2]*m[1];
                fprintf(out,"%.4lf\t",data[r]);
            }
            fprintf(out,"\n");
        }
    fclose(out);
}

void qmult (double *qtin1, double *qtin2, double *qtout, int ptt=0){
    double A=qtin1[0],At=qtin2[0];
    double B=qtin1[1],Bt=qtin2[1];
    double C=qtin1[2],Ct=qtin2[2];
    double Dq=qtin1[3],Dt=qtin2[3];
    (void)ptt;
    qtout[0] = A*At - B*Bt - C*Ct - Dq*Dt;
    qtout[1] = A*Bt + B*At + C*Dt - Dq*Ct;
    qtout[2] = A*Ct + C*At + Dq*Bt - B*Dt;
    qtout[3] = A*Dt + Dq*At + B*Ct - C*Bt;
}

void rotvecq (double *qtn, double *vin, double *vout){
    double qtmp[4];
    double qtmp2[4];
    qtmp2[0]=qtn[0];
    qtmp2[1]=-qtn[1];
    qtmp2[2]=-qtn[2];
    qtmp2[3]=-qtn[3];
    qmult(vin, qtmp2, qtmp);
    qmult(qtn, qtmp, vout);
}

void normq (double *qtnl){
    double mag = 0.0;
    for(int ii=0; ii<4; ii++) mag += qtnl[ii]*qtnl[ii];
    mag = sqrt(mag);
    for(int ii=0; ii<4; ii++) qtnl[ii] /= mag;
}

static std::string shell_quote(const std::string& s)
{
    std::string out = "'";
    for(size_t ii=0; ii<s.size(); ii++){
        if(s[ii]=='\'') out += "'\"'\"'";
        else out += s[ii];
    }
    out += "'";
    return out;
}

static std::string path_dirname(const std::string& path)
{
    size_t pos = path.find_last_of("/\\");
    if(pos==std::string::npos) return ".";
    if(pos==0) return path.substr(0, 1);
    return path.substr(0, pos);
}

static bool write_proteins_raw(const char *fname)
{
    FILE *out = fopen(fname, "wb");
    if(out==NULL) return false;
    const size_t total = static_cast<size_t>(strn) * M;
    std::vector<float> pbuf(total);
    for(int jj=0; jj<3; jj++){
        double **src = (jj==0) ? prott1 : ((jj==1) ? prott2 : prott3);
        for(int ii=0; ii<strn; ii++)
            for(int r=0; r<M; r++)
                pbuf[static_cast<size_t>(ii) * M + r] = static_cast<float>(src[ii][r]);
        if(fwrite(pbuf.data(), sizeof(float), total, out) != total){
            fclose(out);
            return false;
        }
    }
    fclose(out);
    return true;
}

static int write_proteins_h5_with_helper(const char *argv0)
{
    const std::string exe_dir = path_dirname(std::string(argv0));
    const std::string helper = exe_dir + "/tools/protein_preview_to_h5.py";
    const std::string raw = "proteins_preview.raw";
    if(!write_proteins_raw(raw.c_str())){
        fprintf(stderr, "protein_preview: failed to write temporary raw protein dump %s\n", raw.c_str());
        return 1;
    }

    std::ostringstream cmd;
    cmd << "/bin/bash -lc ";
    std::ostringstream inner;
    inner << "if [ -n \"${PROTEIN_PREVIEW_PYTHON:-}\" ]; then "
          << "\"$PROTEIN_PREVIEW_PYTHON\" "
          << shell_quote(helper)
          << " proteins_preview.raw proteins.h5 proteins.xdmf "
          << m[0] << " " << m[1] << " " << m[2] << " " << strn
          << " " << D[0] << " " << D[1] << " " << D[2]
          << "; elif command -v python >/dev/null 2>&1; then python "
          << shell_quote(helper)
          << " proteins_preview.raw proteins.h5 proteins.xdmf "
          << m[0] << " " << m[1] << " " << m[2] << " " << strn
          << " " << D[0] << " " << D[1] << " " << D[2]
          << "; elif command -v python3 >/dev/null 2>&1; then python3 "
          << shell_quote(helper)
          << " proteins_preview.raw proteins.h5 proteins.xdmf "
          << m[0] << " " << m[1] << " " << m[2] << " " << strn
          << " " << D[0] << " " << D[1] << " " << D[2]
          << "; else exit 127; fi";
    cmd << shell_quote(inner.str());

    int status = system(cmd.str().c_str());
    if(status==0) remove(raw.c_str());
    else fprintf(stderr, "protein_preview: HDF5 export helper failed. Rerun with --vtk if you want VTK fallback.\n");
    return status;
}

#ifndef PROTEIN_HEADER
#define PROTEIN_HEADER "protein_arcs.h"
#endif
#include PROTEIN_HEADER

int main(int argc, char *argv[])
{
    char finc[256];
    bool force_vtk = false;
    std::vector<std::string> positional;
    for(int ai=1; ai<argc; ai++){
        if(strcmp(argv[ai], "--vtk")==0) force_vtk = true;
        else positional.push_back(argv[ai]);
    }
    if(positional.size()>0) sprintf(finc,"%s",positional[0].c_str());
    else sprintf(finc,"input.dat");

    printf("protein_preview: starting\n");
    printf("protein_preview: reading main input from %s\n", finc);

    Px0 = (double *) malloc(strn*sizeof(double));
    Py0 = (double *) malloc(strn*sizeof(double));
    Pz0 = (double *) malloc(strn*sizeof(double));
    Pth0 = (double *) malloc(strn*sizeof(double));
    protein_Pth_replica = (double *) malloc(strn*sizeof(double));
    for(int ii=0; ii<strn; ii++){
        Px0[ii]=0.0; Py0[ii]=0.0; Pz0[ii]=0.0; Pth0[ii]=0.0;
        protein_Pth_replica[ii]=0.0;
    }

    if(!fexist(finc)) fail_read(finc, "input file not found");
    if(looks_like_keyed_input(finc)) parse_keyed_input(finc);
    else parse_legacy_input(finc);

    if(fexist("prot_input.dat")){
        printf("protein_preview: reading protein input from prot_input.dat\n");
        parse_keyed_protein_input("prot_input.dat");
    } else {
        printf("protein_preview: prot_input.dat not found, using built-in/default protein settings where needed\n");
    }
    for(int ii=0; ii<strn; ii++) protein_Pth_replica[ii] = Pth;
    protein_have_Pth_file = 0;
    if(fexist("Pth")){
        printf("protein_preview: reading per-replica shell extents from Pth\n");
        FILE *in = fopen("Pth","r");
        for(int ii=0; ii<strn; ii++){
            if(fscanf(in,"%lf",&protein_Pth_replica[ii])!=1) fail_read("Pth", "expected 1 numeric column per line");
        }
        fclose(in);
        protein_have_Pth_file = 1;
    } else {
        printf("protein_preview: Pth not found, using shared Pth=%lf for all replicas\n", Pth);
    }
    validate_protein_settings("prot_input.dat");
    if(fexist("P0s")){
        printf("protein_preview: reading protein positions from P0s\n");
        FILE *in = fopen("P0s","r");
        for(int ii=0; ii<strn; ii++){
            if(fscanf(in,"%lf %lf %lf %lf",&Px0[ii],&Py0[ii],&Pz0[ii],&Pth0[ii])!=4) fail_read("P0s", "expected 4 numeric columns per line");
        }
        fclose(in);
    } else {
        printf("protein_preview: P0s not found, using P0 defaults from input\n");
    }

    double qtn[strn][4];
    for(int ii=0; ii<strn; ii++){
        qtn[ii][0]=1.0; qtn[ii][1]=0.0; qtn[ii][2]=0.0; qtn[ii][3]=0.0;
    }
    if(fexist("qtns")){
        printf("protein_preview: reading quaternions from qtns\n");
        FILE *in = fopen("qtns","r");
        for(int ii=0; ii<strn; ii++){
            if(fscanf(in,"%lf %lf %lf %lf",&qtn[ii][0],&qtn[ii][1],&qtn[ii][2],&qtn[ii][3])!=4) fail_read("qtns", "expected 4 numeric columns per line");
        }
        fclose(in);
    } else {
        printf("protein_preview: qtns not found, using identity quaternions\n");
    }

    if(positional.size()>1) Py = atof(positional[1].c_str());

    M = m[0]*m[1]*m[2];
    printf("protein_preview: grid is %d x %d x %d (M=%d)\n", m[0], m[1], m[2], M);
    printf("protein_preview: using header %s\n", PROTEIN_HEADER);
    malloc2d(&prott1, strn, M);
    malloc2d(&prott2, strn, M);
    malloc2d(&prott3, strn, M);
    printf("protein_preview: allocated protein field arrays for %d replicas\n", strn);

    int doiis[strn];
    for(int ii=0; ii<strn; ii++) doiis[ii]=1;
    double chi[3] = {1.0, 1.0, 1.0};

    printf("protein_preview: building protein fields\n");
    for(int ii=0; ii<strn; ii++){
        if(ii==0 || ((ii+1)%8)==0 || ii==(strn-1))
            printf("protein_preview: building replica %d / %d\n", ii+1, strn);
        mkprot(0, chi, qtn[ii], doiis, ii, -1);
    }

    if(!force_vtk){
        printf("protein_preview: writing proteins.h5 and proteins.xdmf\n");
        int h5_status = write_proteins_h5_with_helper(argv[0]);
        if(h5_status!=0) return 1;
    } else {
        char outfl[128];
        printf("protein_preview: writing VTK outputs\n");
        for(int ii=0; ii<strn; ii++){
            if(ii==0 || ((ii+1)%8)==0 || ii==(strn-1))
                printf("protein_preview: writing replica %d / %d\n", ii+1, strn);
            if(!(pro1==0)){
                sprintf(outfl,"prot1_%d.vtk",ii);
                tovtk(outfl, m, D, prott1[ii]);
            }
            if(!(pro2==0)){
                sprintf(outfl,"prot2_%d.vtk",ii);
                tovtk(outfl, m, D, prott2[ii]);
            }
            if(!(pro3==0)){
                sprintf(outfl,"prot3_%d.vtk",ii);
                tovtk(outfl, m, D, prott3[ii]);
            }
        }
    }

    printf("Protein preview written for %d replicas using %s.\n", strn, PROTEIN_HEADER);
    return 0;
}
