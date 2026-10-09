// scft.cu
//--------------------------------------------------------------
// Main driver for the lipid-water-protein SCFT calculation.
//
// Physics:
// - AB diblock copolymer = lipid
// - homopolymer solvent = water-like component
// - externally imposed protein fields enter the propagator solve
//
// Numerics:
// - pseudo-spectral propagator stepping on the GPU
// - MPI distributes string replicas across ranks
// - `solve_field0()` relaxes the pressure-like/incompressibility field
// - `solve_field()` updates the two composition-like fields and the string
//
// Global array conventions:
// - m[i] = number of mesh points in dimension i
// - M = m[0]*m[1]*m[2] real-space mesh size
// - Mk = m[0]*m[1]*(m[2]/2+1) Fourier-space mesh size
// - NA, NB = contour steps along A and B blocks
// - dsA, dsB = contour step sizes
//--------------------------------------------------------------
#include<math.h>
#include<stdlib.h>
#include<stdio.h>
//#include<fftw3.h>
#include<complex>
#include<assert.h>
#include"mpi.h"
#include<signal.h>
#include<sys/stat.h>
#include<sys/types.h>
#include<errno.h>
#include<fstream>
#include<sstream>
#include<string>
#include<unordered_map>
#include<vector>
#include<algorithm>
#include<cctype>


#define DIM 2 // max number of histories in Anderson = DIM-1

using namespace std;

#ifndef SCFT_STRN
#define SCFT_STRN 48
#endif
const int strn=SCFT_STRN; //strst is # of steps on string. Move to input

int m[3], M, Mk, NA, NB;
double dsA, dsB, pi=4*atan(1.0), phidb, D[3], *F,alpha=0.1,phis[6];
double Pr, Py, Pth, *Px0, *Py0, *Pz0, *Pth0;
double *protein_Pth_replica=NULL;
double protein_Prx=0.24;
double pro1=2.0, pro2=0.0, pro3=0.4, protein_mvin=0.48;
double protein_pitch=0.0;
double protein_patch_wx=0.209;
double protein_patch_wy=0.253;
double protein_patch_offset2=0.0;
double protein_patch_offset3=0.0;
int protein_enabled=1, protein_movable=1;
int protein_hydrophilic_mode=0; // 0=surface, 1=patch
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
int checkpoint_precision=32;
int max_outer=1000;
int field_iterations=100;
int justFE=0;
int dostring=1;
int perp_update=1;
int dangle=0;
int ens=2;
int fix[2]={0,0};
double dangle_Delsum0=0.0;
double ***DEV, ***DDEV, ***WIN;
double *expKA, *expKB, *expKA2, *expKB2;
double ***q1, ***q2, **W, **prott1, **prott2, **prott3, *cubes;// ***cubes;
double **dxprott1, **dxprott2, **dxprott3;
double **dyprott1, **dyprott2, **dyprott3;
double **dzprott1, **dzprott2, **dzprott3;
double **dFdR;
double **Fxfield,**Fyfield,**Fzfield;
double **tors, **forces;


double **phiA, **phiB, **phih, **phicB, **Wp, **Wm, **phip, **phim;
int springsteps=0;
int splineChunkM=0;
FILE *in, *out;
double **dWda,**dSCFTWN, *cppara, *cpperp;//, **Wnew;
double zh=0.142134;
char tms[50];
time_t time0, time1;
//int sigg=0;
volatile sig_atomic_t sigg = 0;

double *FEs0,*FEs1,*FEs2,*FEs,*phctot;

//cube shit
#define DEV_CUBES(r,j,ii) dev_cubes[ii + j*_strn + _strn*4*r]
#define CUBES(r,j,ii) cubes[ii + j*strn + strn*4*r]


double* cube_h;//new double[n];
double* cube_A;//new double[n];
double* cube_l;//new double[npnt];
double* cube_u;//new double[npnt];
double* cube_z;//new double[npnt];
double* cube_c;//new double[npnt];
double* cube_b;//new double[n];
double* cube_d;//new double[n];
double* cube_a;//new double[n];

//cuda:
#include "density.h"
#include "hdf5_io.h"

double FreeE(double **W, const double *chi, const double f, density *D2, double *alf, int *doiis, int *ndo, int *whois, int procid, int numprocs, int typee);
void write_hdf5_outputs(double **W, density *D2, double f, int procid);

typedef std::unordered_map<std::string, std::vector<double>> ParamMap;



//==============================================================
// Parse D-HH:MM:SS or HH:MM:SS or MM:SS into seconds.
//--------------------------------------------------------------
static inline int parse_timeleft(const char *s) {
    int d=0,h=0,m=0; double sec=0;
    if (strchr(s,'-')) {
        if (sscanf(s, "%d-%d:%d:%lf", &d,&h,&m,&sec) != 4) return -1;
    } else {
        int col=0; for (const char *p=s; *p; ++p) if (*p==':') ++col;
        if (col==2) { if (sscanf(s, "%d:%d:%lf", &h,&m,&sec) != 3) return -1; }
        else if (col==1) { if (sscanf(s, "%d:%lf", &m,&sec) != 2) return -1; }
        else return -1;
    }
    long tot = d*86400L + h*3600L + m*60L + (long)sec;
    return tot < 0 ? -1 : (int)tot;
}
//==============================================================
// Query Slurm for remaining wall time on the local rank.
//--------------------------------------------------------------
static inline int slurm_seconds_left_local(void) {
    const char *jobid = getenv("SLURM_JOB_ID");
    if (!jobid || !*jobid) return -1;

    char cmd[256], buf[256] = {0};
    FILE *fp;

    // Try squeue first
    snprintf(cmd, sizeof(cmd), "squeue -h -j %s -o %%L", jobid);
    fp = popen(cmd, "r");
    if (fp) {
        if (fgets(buf, sizeof(buf), fp)) { pclose(fp); buf[strcspn(buf,"\r\n")] = 0; return parse_timeleft(buf); }
        pclose(fp);
    }

    // Fallback to scontrol
    snprintf(cmd, sizeof(cmd), "scontrol show job %s", jobid);
    fp = popen(cmd, "r");
    if (!fp) return -1;
    size_t n = fread(buf, 1, sizeof(buf)-1, fp);
    pclose(fp);
    buf[n] = 0;
    char *p = strstr(buf, "TimeLeft=");
    if (!p) return -1;
    p += 9;
    char val[32] = {0};
    sscanf(p, "%31s", val);
    return parse_timeleft(val);
}

//==============================================================
// Periodically check remaining job time and trigger a graceful stop.
//--------------------------------------------------------------
static inline void poll_time_and_set_sigg_local(int grace_seconds) {
        static time_t last_check = 0;
        time_t now = time(NULL);
        // only check every 30 s (or 60 s)
        if (now - last_check < 120) return;
        last_check = now;

        int left = slurm_seconds_left_local();
        if (left >= 0 && left <= grace_seconds) sigg = 7;
}


//==============================================================
//==============================================================
// Quaternion helpers for rigid-body protein rotation.
//--------------------------------------------------------------
//multiply quaternions using distributive property and
//i^2=j^2=k^2=-1
//ij=k,  ki=j,  jk=i
//ji=-k, ik=-j, kj=-i //anticommutative
// (A + Bi + Cj + Dk)(At + Bti + Cti + Dti)
//= (A*At - B*Bt - C*Ct - D*Dt) + i(A*Bt + B*At + C*Dt - D*Ct) + j(A*Ct + C*At + D*Bt - B*Dt) + k(A*Dt + D*At + B*Ct - C*Bt)
void qmult (double *qtin1, double *qtin2, double *qtout, int ptt=0){
    double m,i,j,k;
    double A=qtin1[0],At=qtin2[0];
    double B=qtin1[1],Bt=qtin2[1];
    double C=qtin1[2],Ct=qtin2[2];
    double D=qtin1[3],Dt=qtin2[3];
    
    if(ptt==1) printf("mt int: %lf %lf %lf %lf\n",At,Bt,Ct,Dt); //for debugging
    
    m = A*At - B*Bt - C*Ct - D*Dt;
    i = A*Bt + B*At + C*Dt - D*Ct;
    j = A*Ct + C*At + D*Bt - B*Dt;
    k = A*Dt + D*At + B*Ct - C*Bt;
    if(ptt==1) printf("mt out: %lf %lf %lf %lf\n",m,i,j,k);
    
    qtout[0]=m;qtout[1]=i;qtout[2]=j;qtout[3]=k;
    
}
// Rotate a 3-vector written as the 4-vector (0,x,y,z) by quaternion qtn:
// vout = qtn vin qtn^-1
void rotvecq (double *qtn, double *vin, double *vout){
    
    double qtmp[4];
    double qtmp2[4]; //inverse of qtn
    qtmp2[0]=qtn[0];
    qtmp2[1]=-qtn[1];
    qtmp2[2]=-qtn[2];
    qtmp2[3]=-qtn[3];
    
    qmult(vin, qtmp2, qtmp);
    qmult(qtn, qtmp, vout);
}

// Normalize a quaternion to limit drift from repeated updates.
void normq (double *qtnl){
    double mag = qtnl[0]*qtnl[0];
    mag += qtnl[1]*qtnl[1];
    mag += qtnl[2]*qtnl[2];
    mag += qtnl[3]*qtnl[3];
    
    mag = sqrt(mag);
    
    qtnl[0] /= mag;
    qtnl[1] /= mag;
    qtnl[2] /= mag;
    qtnl[3] /= mag;
}
// Initialize a quaternion representing a rotation by theta about one axis.
void qinit (double *qtnl, double theta, int axis){
    assert(axis<4 && axis>0);
    qtnl[0]=cos(theta);
    qtnl[1]=0;
    qtnl[2]=0;
    qtnl[3]=0;
    qtnl[axis]=sin(theta);
}

//==============================================================
// restricts a number to between \pm of a value
//--------------------------------------------------------------
double cap (double in, double maxx){
    double signn=1;
    if(in<0) signn=-1; //sign of input
    double ain=fabs(in); //absolute value of input
    return signn*fmin(ain,maxx);
}


//==============================================================
// counts number of lines in a file
//--------------------------------------------------------------
int lines(const char * fname){
    int lines=0;
    FILE *fp;
    fp = fopen(fname,"r");
    if(fp==NULL) return -1;
    while(!feof(fp))
    {
        char ch = fgetc(fp);
        if(ch == '\n')
        {
            lines++;
        }
    }
    fclose(fp);
    return lines;
}
//==============================================================
// counts number of columns in the first non-empty line of a file
//--------------------------------------------------------------
int file_columns(const char *fname){
    FILE *fp = fopen(fname, "r");
    if(fp==NULL) return -1;
    char line[4096];
    int ncols = -1;
    while(fgets(line, sizeof(line), fp) != NULL){
        double a, b, c, d;
        int count = sscanf(line, "%lf %lf %lf %lf", &a, &b, &c, &d);
        if(count > 0){
            ncols = count;
            break;
        }
    }
    fclose(fp);
    return ncols;
}
//==============================================================
// aborts with a clear file-read error
//--------------------------------------------------------------
void fail_read(int procid, const char *fname, const char *detail){
    fprintf(stderr, "Fatal read error (procid=%d) in '%s': %s\n", procid, fname, detail);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
}
//==============================================================
// choose a conservative spline chunk size based on current GPU memory
//--------------------------------------------------------------
int chooseSplineChunkM(int M, int procid){
    size_t freeBytes = 0, totalBytes = 0;
    cudaError_t err = cudaMemGetInfo(&freeBytes, &totalBytes);
    if(err != cudaSuccess){
        if(procid==0) printf("Warning: cudaMemGetInfo failed; using fallback splineChunkM.\n");
        return std::min(M, 200000);
    }

    const double safetyFrac = 0.20; // only use a conservative slice of currently free memory
    const size_t usableBytes = static_cast<size_t>(freeBytes * safetyFrac);
    const size_t bytesPerPoint = static_cast<size_t>(strn) * sizeof(double) * 6; // input + coeffs + eval
    int chunkM = static_cast<int>(usableBytes / bytesPerPoint);
    if(chunkM < 1) chunkM = 1;
    if(chunkM > M) chunkM = M;

    if(procid==0){
        printf("Spline chunks: free GPU memory=%zu bytes, using splineChunkM=%d.\n", freeBytes, chunkM);
    }
    return chunkM;
}
//==============================================================
// keyed-input helpers
//--------------------------------------------------------------
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

ParamMap readParameters(const char *fname, int procid){
    ParamMap params;
    std::ifstream file(fname);
    if(!file.is_open()) fail_read(procid, fname, "failed to open keyed input file");
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
            fail_read(procid, fname, detail);
        }
        std::string key = trim_copy(line.substr(0, eqPos));
        std::string rhs = trim_copy(line.substr(eqPos+1));
        if(key.empty()){
            char detail[256];
            sprintf(detail, "keyed input line %d has an empty key", lineno);
            fail_read(procid, fname, detail);
        }
        std::vector<double> values;
        if(!parse_numeric_tokens(rhs, values)){
            char detail[256];
            sprintf(detail, "keyed input line %d contains a non-numeric value", lineno);
            fail_read(procid, fname, detail);
        }
        if(values.empty()){
            char detail[256];
            sprintf(detail, "keyed input line %d has no numeric values", lineno);
            fail_read(procid, fname, detail);
        }
        params[key] = values;
    }
    return params;
}

int as_int_checked(double value, int procid, const char *fname, const char *key){
    double rounded = floor(value + 0.5);
    if(fabs(value-rounded) > 1E-9){
        char detail[256];
        sprintf(detail, "key '%s' must contain an integer value", key);
        fail_read(procid, fname, detail);
    }
    return static_cast<int>(rounded);
}

const std::vector<double>& require_param(const ParamMap& params, const char *fname, int procid, const char *key, size_t expected_count){
    ParamMap::const_iterator it = params.find(key);
    if(it == params.end()){
        char detail[256];
        sprintf(detail, "required key '%s' is missing", key);
        fail_read(procid, fname, detail);
    }
    if(it->second.size() != expected_count){
        char detail[256];
        sprintf(detail, "key '%s' expected %lu values but found %lu", key, (unsigned long)expected_count, (unsigned long)it->second.size());
        fail_read(procid, fname, detail);
    }
    return it->second;
}

bool get_optional_param(const ParamMap& params, const char *key, size_t expected_count, std::vector<double>& values){
    ParamMap::const_iterator it = params.find(key);
    if(it == params.end()) return false;
    if(it->second.size() != expected_count) return false;
    values = it->second;
    return true;
}

bool get_optional_int_param(const ParamMap& params, const char *key, int &value, int procid, const char *fname)
{
    ParamMap::const_iterator it = params.find(key);
    if(it == params.end()) return false;
    if(it->second.size() != 1){
        char detail[256];
        sprintf(detail, "key '%s' expected 1 value but found %lu", key, (unsigned long)it->second.size());
        fail_read(procid, fname, detail);
    }
    value = as_int_checked(it->second[0], procid, fname, key);
    return true;
}

void parse_keyed_input(const char *fname, int procid, double *chi, double &f, double &Vc, int &N, int *m, double *D, double &Pr, double &Py, double &Pth, double &Px00, double &Py00, double &Pz00, double &Pth00, int &flag, bool &have_protein_lines){
    ParamMap params = readParameters(fname, procid);
    have_protein_lines = false;
    const std::vector<double>& chi_vals = require_param(params, fname, procid, "chi", 3);
    chi[0] = chi_vals[0];
    chi[1] = chi_vals[1];
    chi[2] = chi_vals[2];
    f = require_param(params, fname, procid, "f", 1)[0];
    Vc = require_param(params, fname, procid, "Vc", 1)[0];
    N = as_int_checked(require_param(params, fname, procid, "N", 1)[0], procid, fname, "N");
    const std::vector<double>& m_vals = require_param(params, fname, procid, "m", 3);
    m[0] = as_int_checked(m_vals[0], procid, fname, "m");
    m[1] = as_int_checked(m_vals[1], procid, fname, "m");
    m[2] = as_int_checked(m_vals[2], procid, fname, "m");
    const std::vector<double>& D_vals = require_param(params, fname, procid, "D", 3);
    D[0] = D_vals[0];
    D[1] = D_vals[1];
    D[2] = D_vals[2];
    flag = as_int_checked(require_param(params, fname, procid, "readwin", 1)[0], procid, fname, "readwin");
    get_optional_int_param(params, "justFE", justFE, procid, fname);
    get_optional_int_param(params, "max_outer", max_outer, procid, fname);
    get_optional_int_param(params, "field_iterations", field_iterations, procid, fname);
    get_optional_int_param(params, "dostring", dostring, procid, fname);
    get_optional_int_param(params, "perp_update", perp_update, procid, fname);
    get_optional_int_param(params, "dangle", dangle, procid, fname);
    get_optional_int_param(params, "ens", ens, procid, fname);
    get_optional_int_param(params, "checkpoint_precision", checkpoint_precision, procid, fname);
    if(!(checkpoint_precision==32 || checkpoint_precision==64)){
        fail_read(procid, fname, "checkpoint_precision must be 32 or 64");
    }
    std::vector<double> fix_vals;
    if(get_optional_param(params, "fix", 2, fix_vals)){
        fix[0] = as_int_checked(fix_vals[0], procid, fname, "fix");
        fix[1] = as_int_checked(fix_vals[1], procid, fname, "fix");
    }
    std::vector<double> p_vals, p0_vals;
    bool hasP = get_optional_param(params, "P", 3, p_vals);
    bool hasP0 = get_optional_param(params, "P0", 4, p0_vals);
    if(hasP != hasP0){
        fail_read(procid, fname, "deprecated keyed protein keys must include both 'P' and 'P0' if either is present");
    }
    if(hasP){
        Pr = p_vals[0];
        Py = p_vals[1];
        Pth = p_vals[2];
        Px00 = p0_vals[0];
        Py00 = p0_vals[1];
        Pz00 = p0_vals[2];
        Pth00 = p0_vals[3];
        have_protein_lines = true;
    }
}

void parse_legacy_input(const char *fname, int procid, double *chi, double &f, double &Vc, int &N, int *m, double *D, double &Pr, double &Py, double &Pth, double &Px00, double &Py00, double &Pz00, double &Pth00, int &flag, bool &have_protein_lines){
    std::ifstream file(fname);
    if(!file.is_open()) fail_read(procid, fname, "failed to open legacy input file");
    std::string line;
    std::vector< std::vector<double> > rows;
    while(std::getline(file, line)){
        size_t commentPos = line.find('#');
        if(commentPos != std::string::npos) line.erase(commentPos);
        line = trim_copy(line);
        if(line.empty()) continue;
        std::vector<double> values;
        if(!parse_numeric_tokens(line, values)){
            fail_read(procid, fname, "legacy input contains a non-numeric value");
        }
        rows.push_back(values);
    }
    have_protein_lines = false;
    if(!(rows.size()==4 || rows.size()==6)){
        fail_read(procid, fname, "legacy input must contain either 4 numeric lines (no embedded protein data) or 6 numeric lines (with embedded protein data)");
    }
    if(rows[0].size()!=5) fail_read(procid, fname, "legacy line 1 must contain 5 values: chi0 chi1 chi2 f Vc");
    if(rows[1].size()!=4) fail_read(procid, fname, "legacy line 2 must contain 4 values: N m0 m1 m2");
    if(rows[2].size()!=3) fail_read(procid, fname, "legacy line 3 must contain 3 values: D0 D1 D2");
    chi[0]=rows[0][0]; chi[1]=rows[0][1]; chi[2]=rows[0][2]; f=rows[0][3]; Vc=rows[0][4];
    N = as_int_checked(rows[1][0], procid, fname, "N");
    m[0] = as_int_checked(rows[1][1], procid, fname, "m");
    m[1] = as_int_checked(rows[1][2], procid, fname, "m");
    m[2] = as_int_checked(rows[1][3], procid, fname, "m");
    D[0]=rows[2][0]; D[1]=rows[2][1]; D[2]=rows[2][2];
    if(rows.size()==6){
        if(rows[3].size()!=3) fail_read(procid, fname, "legacy line 4 must contain 3 values: Pr Py Pth");
        if(rows[4].size()!=4) fail_read(procid, fname, "legacy line 5 must contain 4 values: Px00 Py00 Pz00 Pth00");
        if(rows[5].size()!=1) fail_read(procid, fname, "legacy line 6 must contain 1 integer value: flag");
        Pr = rows[3][0];
        Py = rows[3][1];
        Pth = rows[3][2];
        Px00 = rows[4][0];
        Py00 = rows[4][1];
        Pz00 = rows[4][2];
        Pth00 = rows[4][3];
        flag = as_int_checked(rows[5][0], procid, fname, "flag");
        have_protein_lines = true;
    } else {
        if(rows[3].size()!=1) fail_read(procid, fname, "legacy line 4 must contain 1 integer value: flag");
        flag = as_int_checked(rows[3][0], procid, fname, "flag");
    }
}

void parse_keyed_protein_input(const char *fname, int procid, double &Pr, double &Py, double &Pth, double &Px00, double &Py00, double &Pz00, double &Pth00, double &pro1, double &pro2, double &pro3, double &protein_mvin, int &protein_enabled, int &protein_movable){
    ParamMap params = readParameters(fname, procid);
    std::vector<double> values;
    if(get_optional_param(params, "P", 3, values)){
        Pr = values[0];
        Py = values[1];
        Pth = values[2];
    }
    if(get_optional_param(params, "Prx", 1, values)) protein_Prx = values[0];
    if(get_optional_param(params, "P0", 4, values)){
        Px00 = values[0];
        Py00 = values[1];
        Pz00 = values[2];
        Pth00 = values[3];
    }
    if(get_optional_param(params, "pro", 3, values)){
        pro1 = values[0];
        pro2 = values[1];
        pro3 = values[2];
    }
    if(get_optional_param(params, "mvin", 1, values)) protein_mvin = values[0];
    get_optional_int_param(params, "protein_enabled", protein_enabled, procid, fname);
    get_optional_int_param(params, "protein_movable", protein_movable, procid, fname);
    if(get_optional_param(params, "pitch", 1, values)) protein_pitch = values[0];
    if(get_optional_param(params, "patch_wx", 1, values)) protein_patch_wx = values[0];
    if(get_optional_param(params, "patch_wy", 1, values)) protein_patch_wy = values[0];
    if(get_optional_param(params, "patch_offset2", 1, values)) protein_patch_offset2 = values[0];
    if(get_optional_param(params, "patch_offset3", 1, values)) protein_patch_offset3 = values[0];
    get_optional_int_param(params, "hydrophilic_mode", protein_hydrophilic_mode, procid, fname);
    get_optional_int_param(params, "protein_symmetrize", protein_symmetrize, procid, fname);
    get_optional_int_param(params, "protein_pivot_align", protein_pivot_align, procid, fname);
    get_optional_int_param(params, "protein_cap_mode", protein_cap_mode, procid, fname);
    get_optional_int_param(params, "snare_n", snare_n, procid, fname);
    if(get_optional_param(params, "snare_x0", 1, values)) snare_x0 = values[0];
    if(get_optional_param(params, "snare_ring_radius", 1, values)) snare_ring_radius = values[0];
    if(get_optional_param(params, "snare_length", 1, values)) snare_length = values[0];
    if(get_optional_param(params, "snare_radius", 1, values)) snare_radius = values[0];
    if(get_optional_param(params, "snare_pad", 1, values)) snare_pad = values[0];
}

void write_keyed_output(const char *fname, const double *chi, const double f, const double Vc, const int N, const int *m, const double *D, const int flag){
    FILE *fout = fopen(fname, "w");
    if(fout==NULL) return;
    fprintf(fout, "# Main SCFT input / restart file\n");
    fprintf(fout, "chi = %.10g %.10g %.10g    # pairwise interactions for the 3 components\n", chi[0], chi[1], chi[2]);
    fprintf(fout, "f = %.10g                  # diblock A-block fraction\n", f);
    fprintf(fout, "Vc = %.10g                 # copolymer fugacity / concentration control\n", Vc);
    fprintf(fout, "N = %d                     # contour steps along the diblock\n", N);
    fprintf(fout, "m = %d %d %d               # grid points in x y z\n", m[0], m[1], m[2]);
    fprintf(fout, "D = %.10g %.10g %.10g      # box size in x y z\n", D[0], D[1], D[2]);
    fprintf(fout, "readwin = %d               # 0=make initial field, 1=read 3-field wins, 2=read 2-field wins\n", flag);
    fprintf(fout, "max_outer = %d\nfield_iterations = %d\n", max_outer, field_iterations);
    fprintf(fout, "justFE = %d                # 0=run the outer protein/string loop, 1=skip it and only compute free energy\n", justFE);
    fprintf(fout, "dostring = %d              # 0=disable string redistribution/coupling, 1=enable string coupling\n", dostring);
    fprintf(fout, "perp_update = %d           # 0=skip perpendicular string projection, 1=apply it when redistribution starts\n", perp_update);
    fprintf(fout, "dangle = %d                # 0=renormalize string length each redistribution, 1=keep initial redistribution length and move last replica\n", dangle);
    fprintf(fout, "fix = %d %d                # freeze first and/or last replica in solve_field\n", fix[0], fix[1]);
    fprintf(fout, "ens = %d                   # 1=canonical, 2=GC/semi-grand behavior in density evaluation\n", ens);
    fprintf(fout, "checkpoint_precision = %d  # HDF5 restart precision in bits (32 or 64)\n", checkpoint_precision);
    fprintf(fout, "# Protein parameters live in prot_input.dat\n");
    fclose(fout);
}

void validate_main_settings(int procid, const char *source_name, int flag)
{
    if(!(flag==0 || flag==1 || flag==2))
        fail_read(procid, source_name, "readwin must be 0, 1, or 2");
    if(max_outer<0 || field_iterations<1)
        fail_read(procid, source_name, "max_outer must be nonnegative and field_iterations positive");
    if(!(justFE==0 || justFE==1))
        fail_read(procid, source_name, "justFE must be 0 or 1");
    if(!(dostring==0 || dostring==1))
        fail_read(procid, source_name, "dostring must be 0 or 1");
    if(!(perp_update==0 || perp_update==1))
        fail_read(procid, source_name, "perp_update must be 0 or 1");
    if(!(dangle==0 || dangle==1))
        fail_read(procid, source_name, "dangle must be 0 or 1");
    if(!(ens==1 || ens==2))
        fail_read(procid, source_name, "ens must be 1 or 2");
    if(!((fix[0]==0 || fix[0]==1) && (fix[1]==0 || fix[1]==1)))
        fail_read(procid, source_name, "fix values must each be 0 or 1");
}

void write_keyed_protein_input(const char *fname, const double Pr, const double Py, const double Pth, const double Px00, const double Py00, const double Pz00, const double Pth00, const double pro1, const double pro2, const double pro3, const double protein_mvin, const int protein_enabled, const int protein_movable){
    FILE *fout = fopen(fname, "w");
    if(fout==NULL) return;
    fprintf(fout, "# Protein input / restart-adjacent control file\n");
    fprintf(fout, "protein_enabled = %d       # 0=disable external protein fields, 1=enable them\n", protein_enabled);
    fprintf(fout, "protein_movable = %d       # 0=keep protein fixed, 1=move/rotate it from force and torque\n", protein_movable);
    fprintf(fout, "P = %.10g %.10g %.10g      # protein geometry parameters Pr Py Pth\n", Pr, Py, Pth);
    fprintf(fout, "Prx = %.10g                # x-like half-width of the excluded-volume backbone core\n", protein_Prx);
    fprintf(fout, "P0 = %.10g %.10g %.10g %.10g    # fallback protein position/orientation when P0s is absent\n", Px00, Py00, Pz00, Pth00);
    fprintf(fout, "pro = %.10g %.10g %.10g    # strengths of prott1, prott2, prott3\n", pro1, pro2, pro3);
    fprintf(fout, "mvin = %.10g               # radial offset between the two toroidal protein features\n", protein_mvin);
    fprintf(fout, "pitch = %.10g              # axial advance per full turn for helix-like proteins\n", protein_pitch);
    fprintf(fout, "hydrophilic_mode = %d      # 0=surface band, 1=patch-like displaced region\n", protein_hydrophilic_mode);
    fprintf(fout, "patch_wx = %.10g           # x-like half-width for the narrow patch mode\n", protein_patch_wx);
    fprintf(fout, "patch_wy = %.10g           # radial/tangential half-width for the narrow patch mode\n", protein_patch_wy);
    fprintf(fout, "patch_offset2 = %.10g      # angular offset (radians) for prott2 patch mode\n", protein_patch_offset2);
    fprintf(fout, "patch_offset3 = %.10g      # angular offset (radians) for prott3 patch mode\n", protein_patch_offset3);
    fprintf(fout, "protein_cap_mode = %d    # 0=no exterior caps, 1=aligned generic\n", protein_cap_mode);
    fprintf(fout, "protein_symmetrize = %d    # 0=single wrapped object, 1=add z-reflected field, 2=legacy half-copy\n", protein_symmetrize);
    fprintf(fout, "protein_pivot_align = %d   # optional alignment shift for mirrored arc families\n", protein_pivot_align);
    fprintf(fout, "snare_n = %d               # number of snare chunks in the default ring arrangement\n", snare_n);
    fprintf(fout, "snare_x0 = %.10g           # default x position for snare chunks when no snare_positions.dat is present\n", snare_x0);
    fprintf(fout, "snare_ring_radius = %.10g  # ring radius for the default snare arrangement; negative means reuse Py\n", snare_ring_radius);
    fprintf(fout, "snare_length = %.10g       # axial extent of one snare chunk\n", snare_length);
    fprintf(fout, "snare_radius = %.10g       # radial extent of one snare chunk\n", snare_radius);
    fprintf(fout, "snare_pad = %.10g          # extra attraction shell thickness for snare chunks\n", snare_pad);
    fprintf(fout, "# Optional snare_positions.dat may override the default snare ring with explicit local x y z coordinates.\n");
    fclose(fout);
}

void validate_protein_settings(int procid, const char *source_name)
{
    if(!(protein_enabled==0 || protein_enabled==1))
        fail_read(procid, source_name, "protein_enabled must be 0 or 1");
    if(!(protein_movable==0 || protein_movable==1))
        fail_read(procid, source_name, "protein_movable must be 0 or 1");
    if(protein_Prx<=0.0)
        fail_read(procid, source_name, "Prx must be positive");
    if(!(protein_hydrophilic_mode==0 || protein_hydrophilic_mode==1))
        fail_read(procid, source_name, "hydrophilic_mode must be 0 (surface) or 1 (patch)");
    if(!(protein_symmetrize>=0 && protein_symmetrize<=2))
        fail_read(procid, source_name, "protein_symmetrize must be 0 (single), 1 (add), or 2 (legacy copy)");
    if(!(protein_cap_mode==0 || protein_cap_mode==1))
        fail_read(procid, source_name, "protein_cap_mode must be 0 (none) or 1 (aligned)");
    if(!(protein_pivot_align==0 || protein_pivot_align==1))
        fail_read(procid, source_name, "protein_pivot_align must be 0 or 1");
    if(snare_n<=0)
        fail_read(procid, source_name, "snare_n must be positive");
    if(protein_patch_wx<=0.0)
        fail_read(procid, source_name, "patch_wx must be positive");
    if(protein_patch_wy<=0.0)
        fail_read(procid, source_name, "patch_wy must be positive");
    if(snare_length<=0.0)
        fail_read(procid, source_name, "snare_length must be positive");
    if(snare_radius<=0.0)
        fail_read(procid, source_name, "snare_radius must be positive");
    if(snare_pad<0.0)
        fail_read(procid, source_name, "snare_pad must be non-negative");
}
//==============================================================
//==============================================================
// Signal handler for interactive / graceful stop requests.
//--------------------------------------------------------------
void sig_handler(int signo)
{ 
    if (signo == SIGUSR1){
        printf("received SIGUSR1\n");
        sigg=2;
    }
    else if (signo == SIGKILL){
        printf("received SIGKILL\n");
        sigg=3;
    }
    else if (signo == SIGSTOP){
        printf("received SIGSTOP\n");
        sigg=4;
    }
    else if(signo == SIGINT){
        printf("Stop signal received.\n");
        sigg=7;
    }
}

//==============================================================
//==============================================================
// Remove the mean from an array, then shift it by zr0 if requested.
//--------------------------------------------------------------
void avzero(double * A, int n, double zr0=0.0)
{
    double av=0;
    for( int i=0;i<n;i++) av += A[i];
    av /= n;
    for( int i=0;i<n;i++) A[i]-=av;
    //optional zet 0 to something else
    for( int i=0;i<n;i++) A[i]+=zr0;
    
}
//==============================================================
//==============================================================
// Format elapsed wall time since startup into `tms`.
//--------------------------------------------------------------
void tistr()
{
    long int tisec = time(NULL)-time0;
    
    int secs,mins, hours, days;
    
    int lm=60;
    int lh=60*lm;
    int ld=24*lh;
    
    days = tisec/ld;
    hours=(tisec-(days*ld))/lh;
    mins=(tisec-(days*ld)-(hours*lh))/lm;
    secs=tisec-(days*ld)-(hours*lh)-(mins*lm);
    
    if(days>0){
        sprintf(tms,"%d, %02d:%02d:%02d",days,hours,mins,secs);
    } else if(hours>0){
        sprintf(tms,"%d:%02d:%02d",hours,mins,secs);
    } else if(mins>0){
        sprintf(tms,"%d:%02d",mins,secs);
    } else {
        sprintf(tms,"%ds",secs);
    }
    
}

//==============================================================
//calculates the Eucloidian distance between two Ws
//--------------------------------------------------------------
/**/
double EDist (double **W, int ii1, int ii2){
    
    double dist=0;
    // The string distance is defined only on the two composition-like fields.
    // The pressure field is relaxed separately and is not part of the string metric.
    for(int r=0;r<2*M;r++) dist+= pow(W[ii1][r]-W[ii2][r],2.0);
    return sqrt(dist);
}
//==============================================================
// Checks if file exists
//--------------------------------------------------------------
/**/
bool fexist (const char *filename){
    if (FILE * file = fopen(filename, "r"))
    {
        fclose(file);
        return true;
    }
    return false;
}
//==============================================================
// Creates a directory if it does not already exist
//--------------------------------------------------------------
void ensure_dir_exists(const char *dirname){
    if(mkdir(dirname, 0755) != 0 && errno != EEXIST){
        printf("Warning: failed to create directory '%s' (errno=%d).\n", dirname, errno);
    }
}
//==============================================================
// Allocates 2D array
//--------------------------------------------------------------
void malloc2d(double ***ARAY, int l1, int l2){
    *ARAY = (double **)malloc(l1 * sizeof(double *));
    for(int i=0;i<l1;i++) (*ARAY)[i] = (double *)malloc(l2 * sizeof(double));
}
//==============================================================
// Allocates 3D array
//--------------------------------------------------------------
void malloc3d(double ****ARAY, int l1, int l2, int l3){
    *ARAY = (double ***)malloc(l1 * sizeof(double **));
    for(int i=0;i<l1;i++) (*ARAY)[i] = (double **)malloc(l2 * sizeof(double *));
    for(int i=0;i<l1;i++) for(int j=0;j<l2;j++) (*ARAY)[i][j] = (double *)malloc(l3 * sizeof(double));
}
//==============================================================
// output to vtk
//--------------------------------------------------------------
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

//==============================================================
// convert cyl to cartesian and output to vtk
//--------------------------------------------------------------
double xy2th (double x, double y){
    double th = atan2(x,y);
    if(x<0)
        th += 2.0*pi;
    return th;
}
//==============================================================
// Relax the pressure-like / incompressibility field for one replica.
//--------------------------------------------------------------
int solve_field0 (double **W, const double *chi, const double f, density *D2, int ii,
               const int maxIter=1E1, const double errTol=1e-3, int ptt=0)
{
    double lambda=0.05;
    double err=1.0, lnQ, S1, S2;
    int k, r;//, histories,mm,n; //AM stuff
    int skip=299;//,anmix=0;
    double errt=1,dlam1=1.09,dlam2=1.10;//antol=0.00;
    
    for (k=1; k<maxIter && err>errTol; k++) {
        D2->props(W[ii], &lnQ, ii, NA+NB, alpha, phidb, phis, f, k%skip);
        for (r=0; r<M; r++) {
            DEV[ii][0][r]   = 0;
            DEV[ii][0][r+M]   = 0;
            DEV[ii][0][r+2*M] = (chi[0]+chi[1]+chi[2])*(phiB[ii][r]+phiA[ii][r]+phih[ii][r] - 1.0);
            DEV[ii][0][r+2*M] = cap(DEV[ii][0][r+2*M],5.0);
        }
        for (r=0, S1=0.0,S2=0.0; r<3*M; r++) {
            S1 += DEV[ii][0][r]*DEV[ii][0][r];
            S2 += W[ii][r]*W[ii][r];
        }
        err = pow(S1/(M),0.5);
        if(k%skip==0 || !(err==err) || ii==-1 || (ptt==1 && k==1)){
            tistr();
            printf("error0(%d) %5d/%d %.4lE/%.1lE %.2lg (Time: %s)\n",ii,k,maxIter-1,err,errTol,lambda,tms);
            assert(err==err);
        }
        
        // simple mixing
        for (r=0; r<3*M; r++) W[ii][r] = W[ii][r]+lambda*DEV[ii][0][r];
        
        if(err<errt){ //update mixing parameter during simple mixing
            lambda=min(0.01,lambda*dlam1);
        } else {
            lambda=max(0.001,lambda/dlam2);
        }
        errt=err;
    }
    
    return k;
}

//==============================================================
// Main string-method field solve for the composition-like SCFT fields.
// The pressure-like field is still relaxed separately through `solve_field0()`.
//--------------------------------------------------------------
int solve_field (double **W, const double *chi, const double f, density *D2, double *alf, int *doiis, int *ndo, int *whois, int procid, int numprocs, int dostring_local, int doii=-1,
              const int maxIter=1E1, const double errTol=1e-4)
{
    
    double lambda=0.05;
    const bool extra_end_relaxation = true;
    const bool extra_relaxation_only_ends = true;
    //double DEV[strn][DIM][2*M];
    double err=1.0, lnQ, S1, S2,errs[strn], errs3[strn],err3, steps[strn];
    double normDER[strn], normDEV[strn], fitup, fitups[strn];
    int    k, r;//, histories, mm, n;
    int skip=50;//,anmix=0;
    int prints=0;
    double FE;
    double errt=1,dlam1=1.015,dlam2=1.016;//antol=0.00;
    
    double y01 = m[1]/2 + 17;
    double y02 = m[1]/4 - 0;
    double eps=0.1;//,xi;
    char outfl[200];
    double Del[strn-1],Delsum;
    double x[strn],dotWs[strn], dotW,xref[strn],dotWs2[strn];
    int k0=0, kp,ii;
    
    int dospring=1,doredist=0,doperp=0;
    if(doii>-1 || dostring_local!=1) dospring=0,doredist=0,doperp=0;
    if(fix[0]==1) doiis[0]=0;
    if(fix[1]==1) doiis[strn-1]=0;
    
    for(ii=0;ii<strn;ii++){
        errs[ii]=1.0;
        errs3[ii]=0.0;
        steps[ii]=0.0;
        normDER[ii]=0.0;
        normDEV[ii]=0.0;
        fitups[ii]=0.0;
        dotWs[ii]=0.0;
        dotWs2[ii]=0.0;
        x[ii]=0.0;
        xref[ii]=0.0;
    }
    
    for (k=0; k<maxIter && err>errTol; k++) {
        // Every rank must stop on the same iteration: later checkpoint and
        // energy operations contain collectives. Only rank zero queries Slurm.
        if(procid==0) poll_time_and_set_sigg_local(3600);
        int local_stop=sigg, global_stop=0;
        MPI_Allreduce(&local_stop,&global_stop,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
        sigg=global_stop;
        if(k>=springsteps && doii==-1 && dostring_local==1) {
            dospring=0; doredist=1; doperp=perp_update;
        }
        
        // Extra local relaxation can help the string endpoints settle faster.
        if(extra_end_relaxation){
            for(ii=0;ii<strn;ii++){
                const bool is_target = extra_relaxation_only_ends ? (ii==0 || ii==(strn-1)) : true;
                if(doiis[ii]==1 && doii==-1 && is_target && errs[ii]>errTol){
                    for(kp=0;kp<10;kp++){
                        
                        D2->props(W[ii], &lnQ, ii, NA+NB, alpha, phidb, phis, f);
                        for (r=0; r<M; r++){
                            DEV[ii][0][r]   = (chi[0]*(phiB[ii][r] -phiA[ii][r]) +phih[ii][r]*(chi[1]-chi[2])- W[ii][r]);
                            DEV[ii][0][r+M] = (chi[1]*(phih[ii][r]-phiA[ii][r]) + phiB[ii][r]*(chi[0]-chi[2])- W[ii][r+M]);
                            DEV[ii][0][r+2*M] = (chi[0]+chi[1]+chi[2])*(phiB[ii][r]+phiA[ii][r]+phih[ii][r] - 1.0);
                        }
                        for (r=0; r<3*M; r++) W[ii][r] = W[ii][r]+1.0*lambda*DEV[ii][0][r];
                    }
                }
            }
        }
        
        
        for(ii=0;ii<strn;ii++) for (r=0; r<3*M; r++) DEV[ii][0][r]=0;
        k0=0;
        for(ii=0;ii<strn;ii++){
            steps[ii]=0;
            if((doiis[ii]==1 && doii==-1) || doii==ii){ //MPI
                steps[ii]++;
                k0++;
                if(errs[ii]>errTol) steps[ii]+=solve_field0(W, chi, f, D2, ii, 10, errTol); //set to 5
                D2->props(W[ii], &lnQ, ii, NA+NB, alpha, phidb, phis, f);//, k%skip);
                for (r=0; r<M; r++) {
                    DEV[ii][0][r]   = (chi[0]*(phiB[ii][r] -phiA[ii][r]) +phih[ii][r]*(chi[1]-chi[2])- W[ii][r]); //xN_AC-xN_BC
                    DEV[ii][0][r+M] = (chi[1]*(phih[ii][r]-phiA[ii][r]) + phiB[ii][r]*(chi[0]-chi[2])- W[ii][r+M]); //xN_AB-xN_BC
                    DEV[ii][0][r+2*M] = (chi[0]+chi[1]+chi[2])*(phiB[ii][r]+phiA[ii][r]+phih[ii][r] - 1.0);
                    
                    if(ii>0 && ii<strn-1 && dospring==1) DEV[ii][0][r]   += eps*(W[ii+1][r]   + W[ii-1][r]   - 2.0*W[ii][r]);
                    if(ii>0 && ii<strn-1 && dospring==1) DEV[ii][0][r+M] += eps*(W[ii+1][r+M] + W[ii-1][r+M] - 2.0*W[ii][r+M]);
                    
                    
                    //don't adjust too much...
                    DEV[ii][0][r+0*M] = cap(DEV[ii][0][r+0*M],5.0);
                    DEV[ii][0][r+1*M] = cap(DEV[ii][0][r+1*M],5.0);
                    DEV[ii][0][r+2*M] = cap(DEV[ii][0][r+2*M],5.0);
                    
                    
                    // Protein fields are applied inside the propagator solve
                    // rather than being folded into the SCFT fields stored in W.
                    // That keeps the string metric tied to the fields conjugate to
                    // composition, rather than trying to interpolate the protein
                    // geometry itself between neighboring replicas. The tradeoff is
                    // that protein moves can introduce sharper changes in the total
                    // field, so this code uses slow relaxation and extra pressure-
                    // field updates to keep those moves stable.
                }
            }
        }
        
        if(doii==-1){
            Delsum=0;
            for(ii=0;ii<strn-1;ii++){ //distances
                Del[ii]=0;
                Del[ii] = EDist(W, ii, ii+1);
                Delsum+=Del[ii];
            }
            if(dangle==1 && dangle_Delsum0<=0.0) dangle_Delsum0 = Delsum;
            const double Delsum_norm = (dangle==1 && dangle_Delsum0>0.0) ? dangle_Delsum0 : Delsum;
            for(ii=0;ii<strn-1;ii++) Del[ii]/=Delsum_norm; //sum distances
            x[0]=0;
            for(ii=1;ii<strn;ii++) x[ii]=x[ii-1]+Del[ii-1]; //actual positions
            if(dangle==0) x[strn-1]=1; // normal string mode pins the last arclength coordinate
            for(ii=0;ii<strn;ii++) xref[ii] = (1.0*ii)/(strn-1.0); //reference positions
        }
        //fit to function and find derivatives
        if(doperp==1){
            for(ii=0;ii<strn;ii++) if(doiis[ii]==1) for(r=0; r<2*M; r++) dWda[ii][r]=0;
            D2->setSplineX(x);
            for(int r0=0; r0<M; r0+=splineChunkM){
                int chunkSize = min(splineChunkM, M-r0);
                D2->splineDerivsChunked(W, 0, r0, chunkSize, doiis, dWda);
                D2->splineDerivsChunked(W, 1, r0, chunkSize, doiis, dWda);
            }
            
            
            //find normalizations of derivatives = normDER[ii]
            for(ii=0;ii<strn;ii++){
                normDER[ii]=0;
                for(r=0; r<2*M; r++) normDER[ii]+=dWda[ii][r]*dWda[ii][r];  //2Ws (2*M)
                normDER[ii]=sqrt(normDER[ii]);
            }
            
            //find norm of DEV[ii][0][r]  = normDEV[ii]
            for(ii=0;ii<strn;ii++){
                normDEV[ii]=0;
                if(doiis[ii]==1)
                    for(r=0; r<2*M; r++) normDEV[ii]+=DEV[ii][0][r]*DEV[ii][0][r];  //2Ws (2*M)
                normDEV[ii]=sqrt(normDEV[ii]);
            }
            //find dotW[ii] = (DEV[ii][0][r]/normDEV) dot (detivative/normDER)
            for(ii=0;ii<strn;ii++){
                dotWs[ii]=0;
                if(doiis[ii]==1)
                    for(r=0; r<2*M; r++) dotWs[ii] += (DEV[ii][0][r]/normDEV[ii]) * (dWda[ii][r]/normDER[ii]);  //2Ws (2*M)
            }
            //subtract parallel part (dotW[ii]*detivative[ii][r]*normDEV[ii]) from DEV[ii][r]
            for(ii=0;ii<strn;ii++) errs3[ii]=0;
            const int projection_end_ii = (dangle==1) ? strn : strn-1;
            for(ii=1;ii<projection_end_ii;ii++){
                if(doiis[ii]==1 && normDER[ii]>0 && normDEV[ii]>0){
                    for(r=0; r<M; r++) errs3[ii]+= pow(DEV[ii][0][r],2.0)/M;
                    if(k>50 && doperp==1) for(r=0; r<2*M; r++) DEV[ii][0][r] -= (dotWs[ii]/normDER[ii])*dWda[ii][r]*normDEV[ii];  //2Ws (2*M)
                }
            }
            for(ii=0;ii<strn;ii++){
                dotWs2[ii]=0;
                if(doiis[ii]==1)
                    for(r=0; r<2* M; r++) dotWs2[ii] += (DEV[ii][0][r]/normDEV[ii]) * (dWda[ii][r]/normDER[ii]); //2Ws (2*M)
            }
        }
        // At this point DEV contains the field update after any optional
        // spring or perpendicularization corrections.

        //find error(s)
        if(doii==-1){
            for(ii=0;ii<strn;ii++){
                for (r=0, S1=0.0, S2=0.0; r<3*M; r++) {
                    S1 += DEV[ii][0][r]*DEV[ii][0][r];
                    S2 += W[ii][r]*W[ii][r];
                }
                errs[ii] = pow(S1/(3*M),0.5);
            }
        } else {
            for(ii=0;ii<strn;ii++) errs[ii]=0;
            for (r=0, S1=0.0, S2=0.0; r<3*M; r++) {
                S1 += DEV[doii][0][r]*DEV[doii][0][r];
                S2 += W[doii][r]*W[doii][r];
            }
            errs[doii] = pow(S1/(3*M),0.5);
        }
        
        
        
        
        
        // simple mixing with fixed mixing parameter
        for(ii=0;ii<strn;ii++){
            for (r=0; r<3*M; r++) W[ii][r] = W[ii][r]+lambda*DEV[ii][0][r];
        } //ii
        
        
        // Exchange replica state after each update.
        if(numprocs>1){
            for(ii=0;ii<strn;ii++)
                MPI_Bcast(&errs[ii], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
            
            for(ii=0;ii<strn;ii++) MPI_Bcast(&steps[ii], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
            for(ii=0;ii<strn;ii++) MPI_Bcast(&errs3[ii], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
            //   MPI_Barrier(MPI_COMM_WORLD);
            
            
            for(ii=0;ii<strn;ii++)
                MPI_Bcast(W[ii], 3*M, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);  // three SCFT fields
            // MPI_Barrier(MPI_COMM_WORLD);
            
            for(ii=0;ii<strn;ii++)
                MPI_Bcast(&dotWs[ii], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
            // MPI_Barrier(MPI_COMM_WORLD);
            
            for(ii=0;ii<strn;ii++)
                MPI_Bcast(&dotWs2[ii], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
            //  MPI_Barrier(MPI_COMM_WORLD);
            
            for(ii=0;ii<strn;ii++)
                MPI_Bcast(&normDEV[ii], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        dotW=0; for(ii=1;ii<strn-1;ii++) dotW+=dotWs[ii]; //total dotW //maybe find RMS or sum squared or something
        
        //combine errors after exchanges
        err = 0; err3=0;  k0=0; for(ii=0;ii<strn;ii++) {err+= errs[ii]*errs[ii]; errs3[ii] = pow(errs3[ii],0.5); err3+=errs3[ii]; k0 += steps[ii];}
        err=sqrt(err/strn);
        
        // Recompute the arclength parameterization after the field update.
        if(doii==-1){
            Delsum=0;
            for(ii=0;ii<strn-1;ii++){
                Del[ii]=0;
                Del[ii] = EDist(W, ii, ii+1);
                Delsum+=Del[ii];
            }
            
            if(dangle==1 && dangle_Delsum0<=0.0) dangle_Delsum0 = Delsum;
            const double Delsum_norm = (dangle==1 && dangle_Delsum0>0.0) ? dangle_Delsum0 : Delsum;

            //move Ws to fit
            fitup=0;
            for(ii=0;ii<strn-1;ii++) Del[ii]/=Delsum_norm;
            x[0]=0;
            for(ii=1;ii<strn;ii++) x[ii]=x[ii-1]+Del[ii-1];
            if(dangle==0) x[strn-1]=1;
            for(ii=0;ii<strn;ii++) alf[ii]=x[ii];
            for(ii=0;ii<strn;ii++) fitups[ii] = x[ii]-xref[ii];
            const int fitup_end_ii = (dangle==1) ? strn : strn-1;
            for(ii=1;ii<fitup_end_ii;ii++) fitup += pow(fitups[ii],2.0);
            fitup /= (fitup_end_ii - 1.0);
            fitup = sqrt(fitup);
        }
        
        if(doredist==1){
            D2->setSplineX(x);
            D2->setSplineXRef(xref);
            for (int r0=0; r0<M; r0+=splineChunkM){
                int chunkSize = min(splineChunkM, M-r0);
                D2->splineRedistChunked(W, 0, r0, chunkSize, doiis, 0.01, dangle);
                D2->splineRedistChunked(W, 1, r0, chunkSize, doiis, 0.05, dangle);
            }
            if(numprocs>1){
                for(ii=0;ii<strn;ii++)
                    MPI_Bcast(W[ii], 3*M, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }
        //updated
        // */
        
        
        //if(procid==0)
        if(k%skip==0 || k==maxIter-1 || k==-1 || !(err>errTol) || sigg==7) {
            if(procid==0){
                tistr();
                if(doii==-1){
                    if(strn>5){
                        printf("error  %5d/%d %lE %.4lE/%.1lE %.2lg.\tEs: %.2lE %.2lE ... %.2lE ... %.2lE %.2lE... %.2lg.. %.1lf (Time: %s)\n",k,maxIter,dotW,err,errTol,lambda, errs[0],errs[1],errs[strn/2],errs[strn-2],errs[strn-1],fitup,(1.0*k0)/strn,tms);
                        sprintf(outfl,"errors");
                        out = fopen(outfl,"w");
                        for(ii=0;ii<strn;ii++) fprintf(out,"%d %.2lE\n",ii,errs[ii]);
                        fclose(out);
                        
                    } else {
                        printf("error  %5d/%d %lE %.4lE/%.1lE %.2lg.\tEs:",k,maxIter,dotW,err,errTol,lambda);
                        for(ii=0;ii<strn;ii++) printf(" %.2lE",errs[ii]);
                        printf("... %.2lg.. %.1lf (Time: %s)\n",fitup,(1.0*k0)/strn,tms);
                    }

                } else {
                    printf("error[%d]  %5d/%d %.4lE/%.1lE %.2lg.\tE: %.2lE ... %.1lf (Time: %s)\n",doii,k,maxIter,err,errTol,lambda, errs[doii],(1.0*k0),tms);
                }
                assert(err==err);
            }
            prints=prints+1;
        }
        //print configs
        if((k%(skip)==0 && k>0) || k==maxIter-1 || k==-1 || !(err>errTol) || sigg==7){
#ifdef USE_HDF5
            write_hdf5_outputs(W, D2, f, procid);
#else
            for(ii=0;ii<strn;ii++){
                if(((doiis[ii]==1 && doii==-1) || doii==ii) && err==err){
                    sprintf(outfl,"rhoA_%d.vtk",ii);
                    tovtk(outfl, m, D, phiA[ii]);
                    sprintf(outfl,"rhoB_%d.vtk",ii);
                    tovtk(outfl, m, D, phiB[ii]);
                    
                    for(r=0;r<M;r++) phip[ii][r]=phiA[ii][r]+phiB[ii][r] + phih[ii][r];
                    sprintf(outfl,"rhoP_%d.vtk",ii);
//                    tovtk(outfl, m, D, phip[ii]);
                    sprintf(outfl,"win%d",ii);
                    out=fopen(outfl,"w");
                    for (r=0;r<M;r++) fprintf(out,"%.6lf %.6lf %.6lf\n",W[ii][r],W[ii][r+M],W[ii][r+2*M]);
                    fclose(out);
                    
                }
            }
#endif
        }
        if(err<errt){ //update mixing parameter during simple mixing
            lambda=min(0.2,lambda*dlam1);
        } else {
            lambda=max(0.05,lambda/dlam2);
        }
        errt=err;
        
        if(sigg==7) {
            MPI_Barrier(MPI_COMM_WORLD);
            printf("Time's up (procid=%d).\n",procid);
            break;
        }
        
        
    }
    
    return k;
}
//==============================================================
// Initializes lookup table for laplacian operator
//--------------------------------------------------------------
/**/
/**/
void lapin0 (double *expK, double *D, double ds, double fl){
    for (int k0=-(m[0]-1)/2; k0<=m[0]/2; k0++) {
        const int K0 = (k0<0)?(k0+m[0]):k0;
        double A0 = k0*k0/(4.0*D[0]*D[0]);
        for (int k1=-(m[1]-1)/2; k1<=m[1]/2; k1++) {
            const int K1 = (k1<0)?(k1+m[1]):k1;
            double A1 = A0+k1*k1/(4.0*D[1]*D[1]);
            for (int k2=0; k2<=m[2]/2; k2++) {
                double A2 = A1+k2*k2/(4.0*D[2]*D[2]);
                const int k = k2 + (m[2]/2+1)*(K1+m[1]*K0);
                double k_sq = 4*pi*pi*A2;
                expK[k] = exp(-k_sq*ds/(6*fl))/(M);
            }
        }
    }
    
}

//==============================================================
// Initializes lookup table for laplacian operator
//--------------------------------------------------------------
/**/
void lapin (double *expK, double *D, double ds, double fl){
    for (int k0=0; k0<m[0]; k0++) {
        const int K0 = (k0<0)?(k0+m[0]):k0;
        double A0 = k0*k0/(D[0]*D[0]);  //was 4* when periodic
        for (int k1=0; k1<m[1]; k1++) {
            const int K1 = k1;
            double A1 = A0+k1*k1/(D[1]*D[1]);
            for (int k2=0; k2<m[2]; k2++) {
                double A2 = A1+k2*k2/(D[2]*D[2]);
                const int k = K0 + m[0]*K1 + m[0]*m[1]*k2; //2*m[0] because hc
                double k_sq = pi*pi*A2;
                expK[k] = exp(-k_sq*ds/(6*fl))/(8.0*Mk);  //that 8 is 2^number of reflecting boundaries
            }
        }
    }
}

//==============================================================
// Finds free energy
//--------------------------------------------------------------
double FreeE (double **W, const double *chi, const double f, density *D2, double *alf, int *doiis, int *ndo, int *whois, int procid, int numprocs, int typee=0){
    double errTol=1E-4;
    double maxIter=1E2;
    double wa,wb,wh,waa,wba,wha;
    double FE=0;
    int ii;
    double lnQ;
    for(ii=0;ii<strn;ii++){
        const bool owns_fixed_endpoint =
            (fix[0]==1 && ii==0 && procid==whois[0]) ||
            (fix[1]==1 && ii==(strn-1) && procid==whois[strn-1]);
        if(doiis[ii]==1 || owns_fixed_endpoint){
            if(typee==0){
                solve_field0(W, chi, f, D2, ii, maxIter, errTol);
                D2->props(W[ii], &lnQ, ii, NA+NB, alpha, phidb, phis, f);//, k%skip);
            }
            FEs0[ii] = 0;
            FEs0[ii] = -lnQ;
            FEs1[ii]=0;
            FEs2[ii]=0;
            phctot[ii]=0;
            for(int r=0;r<M;r++) {
                wa=(W[ii][r]+W[ii][r+M]+W[ii][r+2*M])/3.0;
                wb=W[ii][r+2*M]+W[ii][r+M] - 2.0*wa;
                wh=W[ii][r+2*M]+W[ii][r  ] - 2.0*wa;
                FEs1[ii] += (chi[0]*phiA[ii][r]*phiB[ii][r]
                             +chi[1]*phiA[ii][r]*phih[ii][r]
                             +chi[2]*phiB[ii][r]*phih[ii][r]
                             -wa*phiA[ii][r]
                             -wb*phiB[ii][r]
                             -wh*phih[ii][r]
                             )/M;
                phctot[ii]+=phiA[ii][r]/(M*f);
                //external field (protein):
                waa= prott1[ii][r]+prott2[ii][r]-prott3[ii][r];//wa = pb*x0+pc*x1+P1+P2-P3+xi
                wba= prott1[ii][r]-prott2[ii][r]+prott3[ii][r];//wb = pa*x0+pc*x2+P1-P2+P3+xi
                wha=-prott1[ii][r]+prott2[ii][r]+prott3[ii][r];//wc = pa*x1+pb*x2-P1+P2+P3+xi
                
                //if(typee==0 || typee==2)
                FEs2[ii] += (waa*phiA[ii][r]
                             +  wba*phiB[ii][r]
                             +  wha*phih[ii][r])/M;
            }
            // `lnQ` already includes the polymer response to the external protein
            // bias through the propagator solve. Because the protein fields are not
            // stored in `W`, they are not part of the usual SCFT double-counting
            // correction. Keep `FEs2` only as a diagnostic contribution here.
            FEs[ii] = FEs0[ii] + FEs1[ii];
            FE+=FEs[ii];
        }
    }
    if(numprocs>1){
        for(ii=0;ii<strn;ii++) MPI_Bcast(&FEs[ii], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
        for(ii=0;ii<strn;ii++) MPI_Bcast(&FEs0[ii], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
        for(ii=0;ii<strn;ii++) MPI_Bcast(&FEs1[ii], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
        for(ii=0;ii<strn;ii++) MPI_Bcast(&FEs2[ii], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
        for(ii=0;ii<strn;ii++) MPI_Bcast(&phctot[ii], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    FE=0;
    for(ii=0;ii<strn;ii++) FE+=FEs[ii];
    return FE;
}

void refresh_view_fields_on_root(double **W, density *D2, double f, int procid)
{
    if(procid!=0) return;
    double lnQ_local = 0.0;
    for(int ii=0; ii<strn; ii++){
        D2->props(W[ii], &lnQ_local, ii, NA+NB, alpha, phidb, phis, f);
        for(int r=0; r<M; r++) phip[ii][r] = phiA[ii][r] + phiB[ii][r] + phih[ii][r];
    }
}

//==============================================================
// Write restart and concentration-view HDF5/XDMF outputs on rank 0.
//--------------------------------------------------------------
void write_hdf5_outputs(double **W, density *D2, double f, int procid)
{
    if(procid!=0) return;
#ifdef USE_HDF5
    // Refresh host-side concentration fields before writing the restart and
    // visualization files. Protein visualization is written from `mkprot()`.
    refresh_view_fields_on_root(W, D2, f, procid);
    if(write_wins_hdf5_atomic("wins.h5", W, m, checkpoint_precision)){
        write_wins_xdmf("wins.xdmf", "wins.h5", m, D, checkpoint_precision);
    }
    if(write_concentrations_hdf5_atomic("concentrations.h5", m, phiA, phiB, phih)){
        write_concentrations_xdmf("concentrations.xdmf", "concentrations.h5", m, D);
        write_concentrations_derived_xdmf("concentrations_derived.xdmf", "concentrations.h5", m, D);
    }
#endif
}

#ifndef PROTEIN_HEADER
#define PROTEIN_HEADER "protein_arcs.h"
#endif
#include PROTEIN_HEADER

//==============================================================
// Build spatial derivatives of the protein fields by centered differences.
//--------------------------------------------------------------
void difprot (int procid, int ii, int times=0){
    int r,rxp,rxm,ryp,rym,rzp,rzm;
    double dp,pp,pm,dx,dy,dz;
    int xp,xm,yp,ym,zp,zm;
    dx=D[0]/m[0];
    dy=D[1]/m[1];
    dz=D[2]/m[2];
    for (int x=0; x<m[0]; x++){
        if(x==0){
            xp=1;
            xm=m[0]-1;
        } else if(x==(m[0]-1)){
            xp=0;
            xm=m[0]-2;
        } else {
            xp=x+1;
            xm=x-1;
        }
        for (int y=0; y<m[1]; y++){
            if(y==0){
                yp=1;
                ym=m[1]-1;
            } else if(y==(m[1]-1)){
                yp=0;
                ym=m[1]-2;
            } else {
                yp=y+1;
                ym=y-1;
            }
            for (int z=0; z<m[2]; z++) {
                if(z==0){
                    zp=1;
                    zm=m[2]-1;
                } else if(z==(m[2]-1)){
                    zp=0;
                    zm=m[2]-2;
                } else {
                    zp=z+1;
                    zm=z-1;
                }
                r = (x*m[1]+y)*m[2]+z;
                rxp = (xp*m[1]+y)*m[2]+z;
                rxm = (xm*m[1]+y)*m[2]+z;
                ryp = (x*m[1]+yp)*m[2]+z;
                rym = (x*m[1]+ym)*m[2]+z;
                rzp = (x*m[1]+y)*m[2]+zp;
                rzm = (x*m[1]+y)*m[2]+zm;
                
                //printf("here %d (%d,%d) %d (%d,%d) %d (%d,%d). %d: %d %d, %d %d, %d %d... %d\n",x,xp,xm,y,yp,ym,z,zp,zm,r,rxp,rxm,ryp,rym,rzp,rzm,M);
                
                ///prott1
                ///ddx
                pp=prott1[ii][rxp]; pm=prott1[ii][rxm]; dp=(pp-pm)/(2.0*dx); dxprott1[ii][r]=dp;
                pp=prott2[ii][rxp]; pm=prott2[ii][rxm]; dp=(pp-pm)/(2.0*dx); dxprott2[ii][r]=dp;
                pp=prott3[ii][rxp]; pm=prott3[ii][rxm]; dp=(pp-pm)/(2.0*dx); dxprott3[ii][r]=dp;
                pp=prott1[ii][ryp]; pm=prott1[ii][rym]; dp=(pp-pm)/(2.0*dy); dyprott1[ii][r]=dp;
                pp=prott2[ii][ryp]; pm=prott2[ii][rym]; dp=(pp-pm)/(2.0*dy); dyprott2[ii][r]=dp;
                pp=prott3[ii][ryp]; pm=prott3[ii][rym]; dp=(pp-pm)/(2.0*dy); dyprott3[ii][r]=dp;
                pp=prott1[ii][rzp]; pm=prott1[ii][rzm]; dp=(pp-pm)/(2.0*dz); dzprott1[ii][r]=dp;
                pp=prott2[ii][rzp]; pm=prott2[ii][rzm]; dp=(pp-pm)/(2.0*dz); dzprott2[ii][r]=dp;
                pp=prott3[ii][rzp]; pm=prott3[ii][rzm]; dp=(pp-pm)/(2.0*dz); dzprott3[ii][r]=dp;
                
            }
        }
    }
}


//==============================================================
// Compute the force density induced by the external protein fields.
// With the protein treated as an imposed external field, the local force
// density is phi * dW/dr.
//--------------------------------------------------------------
void Fpartfield (int *doiis, density *D2, double f){
    double der1,der2,der3;
    double waa,wba,wha;
    double pa,pb,ph,pa0,pb0,ph0;// dp1,dp2,dp10,dp20;
    double lnQ;
    for(int ii=0;ii<strn;ii++){
        if(doiis[ii]==1){
            D2->props(W[ii], &lnQ, ii, NA+NB, alpha, phidb, phis, f);//, k%skip);
            pa0=phiA[ii][M-1]; pb0=phiB[ii][M-1]; ph0=phih[ii][M-1];
            for (int r=0; r<M; r++){
                pa =phiA[ii][r]; pb=phiB[ii][r]; ph=phih[ii][r];
                
                //x
                der1 = dxprott1[ii][r];  //force = -dF/dr = -dF/dW dW/dR =
                der2 = dxprott2[ii][r];
                der3 = dxprott3[ii][r];
                waa= der1+der2-der3;//wa = pb*x0+pc*x1+P1+P2-P3+xi
                wba= der1-der2+der3;//wb = pa*x0+pc*x2+P1-P2+P3+xi
                wha=-der1+der2+der3;//wc = pa*x1+pb*x2-P1+P2+P3+xi
                
                der3=0;
                
                Fxfield[ii][r] = waa*pa+  wba*pb+  wha*ph;
                Fxfield[ii][r]-= waa*pa0+  wba*pb0+  wha*ph0; //subtract off bulk - not necessary but makes fields easier to interprret and help when protein goes over the box limit
                
                //y
                der1 = dyprott1[ii][r];
                der2 = dyprott2[ii][r];
                der3 = dyprott3[ii][r];
                waa= der1+der2-der3;//wa = pb*x0+pc*x1+P1+P2-P3+xi
                wba= der1-der2+der3;//wb = pa*x0+pc*x2+P1-P2+P3+xi
                wha=-der1+der2+der3;//wc = pa*x1+pb*x2-P1+P2+P3+xi
                
                Fyfield[ii][r] = waa*pa+  wba*pb+  wha*ph;
                Fyfield[ii][r]-= waa*pa0+  wba*pb0+  wha*ph0;
                
                //z
                der1 = dzprott1[ii][r];
                der2 = dzprott2[ii][r];
                der3 = dzprott3[ii][r];
                waa= der1+der2-der3;//wa = pb*x0+pc*x1+P1+P2-P3+xi
                wba= der1-der2+der3;//wb = pa*x0+pc*x2+P1-P2+P3+xi
                wha=-der1+der2+der3;//wc = pa*x1+pb*x2-P1+P2+P3+xi
                
                Fzfield[ii][r] = waa*pa+  wba*pb+  wha*ph;
                Fzfield[ii][r]-= waa*pa0+  wba*pb0+  wha*ph0;
            }
        }
    }
    
    
    
}
//FreeE (double **W, const double *chi, const double f, density *D2, double *alf, int *doiis, int *ndo, int *whois, int procid, int numprocs)
void Fpartnum (double **W, double *chi, double **qtnn, const double f, density *D2, double *alf, int *doiis, int *ndo, int *whois, int procid, int numprocs, int typee){
    
    double Pxl, Pyl, Pzl; //local to the subroutine
    double Pxp, Pyp, Pzp;
    double Pxm, Pym, Pzm;
    double dP=0.08;
    double Fp, Fm;//, F0;
    double V = D[0]*D[1]*D[2];
    
    for(int ii=0;ii<strn;ii++){
        Pxl=Px0[ii];
        Pyl=Py0[ii];
        Pzl=Pz0[ii];
        
        Pxp=Pxl+dP;
        Pxm=Pxl-dP;
        Pyp=Pyl+dP;
        Pym=Pyl-dP;
        Pzp=Pzl+dP;
        Pzm=Pzl-dP;
        
        /// x derivative
        Px0[ii]=Pxp;
        mkprot(procid,chi,qtnn[ii],doiis,ii);
        Fp=FreeE(W, chi, f, D2, alf, doiis, ndo, whois, procid, numprocs,typee);
        Px0[ii]=Pxm;
        mkprot(procid,chi,qtnn[ii],doiis,ii);
        Fm=FreeE(W, chi, f, D2, alf, doiis, ndo, whois, procid, numprocs,typee);
        Px0[ii]=Pxl;
        mkprot(procid,chi,qtnn[ii],doiis,ii);
        dFdR[ii][0]=V*(Fp-Fm)/(2.0*dP);
        
        /// y derivative
        Py0[ii]=Pyp;
        mkprot(procid,chi,qtnn[ii],doiis,ii);
        Fp=FreeE(W, chi, f, D2, alf, doiis, ndo, whois, procid, numprocs,typee);
        Py0[ii]=Pym;
        mkprot(procid,chi,qtnn[ii],doiis,ii);
        Fm=FreeE(W, chi, f, D2, alf, doiis, ndo, whois, procid, numprocs,typee);
        Py0[ii]=Pyl;
        mkprot(procid,chi,qtnn[ii],doiis,ii);
        dFdR[ii][1]=V*(Fp-Fm)/(2.0*dP);
        
        /// z derivative
        Pz0[ii]=Pzp;
        mkprot(procid,chi,qtnn[ii],doiis,ii);
        Fp=FreeE(W, chi, f, D2, alf, doiis, ndo, whois, procid, numprocs,typee);
        Pz0[ii]=Pzm;
        mkprot(procid,chi,qtnn[ii],doiis,ii);
        Fm=FreeE(W, chi, f, D2, alf, doiis, ndo, whois, procid, numprocs,typee);
        Pz0[ii]=Pzl;
        mkprot(procid,chi,qtnn[ii],doiis,ii);
        dFdR[ii][2]=V*(Fp-Fm)/(2.0*dP);
    }
}
//==============================================================
// Integrate torque about the protein center from the force-density fields.
//--------------------------------------------------------------
void torque (int ii, double *torques, int axis=0){
    int r;
    double zz,zz2,yy,yy2,xx,xx2;//,rr,rr2;
    double torx=0,tory=0,torz=0;
    for (int x=0; x<m[0]; x++)
        for (int y=0; y<m[1]; y++)
            for (int z=0; z<m[2]; z++) { //initial W here - right now parallel to walls
                r = (x*m[1]+y)*m[2]+z;// PROTEIN // ADD IN
                zz=z*D[2]/m[2]; zz2 = zz-(D[2]/2.0 + Pz0[ii]);
                yy=y*D[1]/m[1]; yy2 = yy-(D[1]/2.0 + Py0[ii]);
                xx=x*D[0]/m[0]; xx2 = xx-(D[0]/2.0 + Px0[ii]);
                torx+=Fyfield[ii][r]*zz2 - Fzfield[ii][r]*yy2;
                tory+=Fzfield[ii][r]*xx2 - Fxfield[ii][r]*zz2;
                torz+=Fxfield[ii][r]*yy2 - Fyfield[ii][r]*xx2;
            }
    torx *= D[0]*D[1]*D[2]/M;
    tory *= D[0]*D[1]*D[2]/M;
    torz *= D[0]*D[1]*D[2]/M;
    torques[0]=torx;
    torques[1]=tory;
    torques[2]=torz;
}
//==============================================================
// Integrate total force from the force-density fields.
//--------------------------------------------------------------
void sumforce (double **forc, int ii){
    double fx=0,fy=0,fz=0;
    
    for (int r=0; r<M; r++){
        fx += Fxfield[ii][r];
        fy += Fyfield[ii][r];
        fz += Fzfield[ii][r];
    }
    
    fx *= D[0]*D[1]*D[2]/M;
    fy *= D[0]*D[1]*D[2]/M;
    fz *= D[0]*D[1]*D[2]/M;
    
    forc[ii][0]= fx;
    forc[ii][1]= fy;
    forc[ii][2]= fz;
}

/**/
//==============================================================
// Key variables in main:
//
// W[16*M] = per-string workspace:
//   [0:M)     field 1
//   [M:2*M)   field 2
//   [2*M:3*M) pressure / incompressibility-like field
//   [3*M:11*M) densities and helper fields populated by density::props()
//   [11*M:16*M) string-method work buffers
// r = (x*m[1]+y)*m[2]+z = array position for (x,y,z)
// FE = free energy
// lnQ = log of the partition function
// chi = chi*N
// f = volume fraction of the A block
// N = number of contour steps 
//
//--------------------------------------------------------------
int main (int argc, char *argv[])
{
    double chi[3], f, FE;//, S1, S2; //D was here //redo Ding if necessary
    int    x, y, z, r, N, flag,ii=0;
    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    
    int seed = (12345+time(NULL));
    srand(seed);
    
    //test gpu
    printf("Hello World from host!\n");
    print_from_gpu<<<1,1>>>();
    cudaDeviceSynchronize();
    
    //sinal catching
    //  if (signal(SIGUSR1, sig_handler) == SIG_ERR)
    //       printf("\ncan't catch SIGUSR1\n");
    //   if (signal(SIGKILL, sig_handler) == SIG_ERR)
    //       printf("\ncan't catch SIGKILL\n");
    //   if (signal(SIGSTOP, sig_handler) == SIG_ERR)
    //       printf("\ncan't catch SIGSTOP\n");
    if (signal(SIGINT, sig_handler) == SIG_ERR)
        printf("\ncan't catch SIGINT\n");
    
    
    //initialize MPI
    int procid=0, numprocs=1,ierr=0;
    //ierr = MPI_Init ( &argc, &argv );
    // /*
    ierr = MPI_Init (NULL,NULL);
    if ( ierr != 0 )
    {
        printf ( "\n" );
        printf ( "HELLO_MPI - Fatal error!\n" );
        printf ( "  MPI_Init returned nonzero IERR.\n" );
        exit ( 1 );
    }
    //initialize mpi
    ierr = MPI_Comm_size ( MPI_COMM_WORLD, &numprocs );
    ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &procid );
    // */
    time0 = time(NULL);
    tistr();
    printf("Hello from procid %d/%d (Time: %s)\n",procid,numprocs,tms);
    char finc[100];
    char outfl[200];
    char winf[100];
    char pouts[100];
    char qouts[100];
    int finint;
    
    if(argc>1)
        sprintf(finc,"%s",argv[1]);
    else
        sprintf(finc,"input.dat");
    
    tistr();
    printf("Reading input from %s (procid=%d, Time: %s).\n",finc,procid,tms);
    
    double Vc,Pth00,Px00,Py00,Pz00;
    bool used_keyed_input = false;
    bool input_has_embedded_protein = false;
    bool have_any_protein_input = false;
    const char *protein_input_name = "prot_input.dat";
    Pth0 = (double *) malloc(strn*sizeof(double ));
    Px0 = (double *) malloc(strn*sizeof(double ));
    Py0 = (double *) malloc(strn*sizeof(double ));
    Pz0 = (double *) malloc(strn*sizeof(double ));
    
    double qtn[strn][4],dqtnx[strn][4],dqtny[strn][4],dqtnz[strn][4]; //define quaternians for rotations
    for(ii=0;ii<strn;ii++){
        qtn[ii][0]=1; qtn[ii][1]=0; qtn[ii][2]=0; qtn[ii][3]=0;
        normq(qtn[ii]);
    }
    
    
    Px00 = Py00 = Pz00 = Pth00 = 0.0;
    Pr = Py = Pth = 0.0;
    if(!fexist(finc)) fail_read(procid, finc, "input file not found");
    if(looks_like_keyed_input(finc)){
        parse_keyed_input(finc, procid, chi, f, Vc, N, m, D, Pr, Py, Pth, Px00, Py00, Pz00, Pth00, flag, input_has_embedded_protein);
        if(procid==0) printf("Detected keyed input format.\n");
        used_keyed_input = true;
    } else {
        parse_legacy_input(finc, procid, chi, f, Vc, N, m, D, Pr, Py, Pth, Px00, Py00, Pz00, Pth00, flag, input_has_embedded_protein);
        if(procid==0) printf("Detected legacy positional input format.\n");
    }
    have_any_protein_input = input_has_embedded_protein;
    if(fexist(protein_input_name)){
        parse_keyed_protein_input(protein_input_name, procid, Pr, Py, Pth, Px00, Py00, Pz00, Pth00, pro1, pro2, pro3, protein_mvin, protein_enabled, protein_movable);
        have_any_protein_input = true;
        if(procid==0) printf("Read protein input from %s.\n", protein_input_name);
    } else if(procid==0){
        printf("Protein input file %s not found; using embedded/default protein settings if available.\n", protein_input_name);
    }
    if(!have_any_protein_input){
        fail_read(procid, protein_input_name, "protein settings are missing; provide prot_input.dat or use an old input file that still contains embedded protein lines");
    }
    validate_main_settings(procid, finc, flag);
    if(procid==0 && !used_keyed_input){
        write_keyed_output("input.dat", chi, f, Vc, N, m, D, flag);
        write_keyed_protein_input(protein_input_name, Pr, Py, Pth, Px00, Py00, Pz00, Pth00, pro1, pro2, pro3, protein_mvin, protein_enabled, protein_movable);
        printf("Wrote keyed restart inputs: input.dat and %s.\n", protein_input_name);
    }
    (void)finint;
    
    sprintf(pouts,"P0s");
    if(fexist(pouts)){
        in = fopen(pouts,"r");
        if(in==NULL) fail_read(procid, pouts, "failed to open P0s");
        for(ii=0;ii<strn;ii++) {
            finint=fscanf(in,"%lf %lf %lf %lf",&Px0[ii],&Py0[ii], &Pz0[ii], &Pth0[ii]);
            if(finint!=4) fail_read(procid, pouts, "expected 4 numeric columns per row for all string replicas");
        }
        fclose(in);
    } else {
        tistr();
        printf("\'%s\' not found; using fallback protein positions from input (procid=%d, Time: %s).\n",pouts,procid,tms);
        for(ii=0;ii<strn;ii++) {
            Px0[ii]=Px00;
            Py0[ii]=Py00;
            Pz0[ii]=Pz00;
            Pth0[ii]=Pth00;
        }
    }

    protein_Pth_replica = (double *) malloc(strn*sizeof(double));
    for(ii=0; ii<strn; ii++) protein_Pth_replica[ii] = Pth;
    protein_have_Pth_file = 0;
    if(fexist("Pth")){
        in = fopen("Pth","r");
        if(in==NULL) fail_read(procid, "Pth", "failed to open Pth");
        for(ii=0; ii<strn; ii++){
            finint=fscanf(in,"%lf",&protein_Pth_replica[ii]);
            if(finint!=1) fail_read(procid, "Pth", "expected 1 numeric column per row for all string replicas");
        }
        fclose(in);
        protein_have_Pth_file = 1;
        if(procid==0) printf("Read per-replica shell extents from Pth.\n");
    } else if(procid==0){
        printf("'Pth' not found; using shared Pth=%lf for all replicas.\n", Pth);
    }
    
    sprintf(qouts,"qtns");
    if(fexist(qouts)){
        in = fopen(qouts,"r");
        if(in==NULL) fail_read(procid, qouts, "failed to open qtns");
        for(ii=0;ii<strn;ii++) {
            finint=fscanf(in,"%lf %lf %lf %lf",&qtn[ii][0],&qtn[ii][1], &qtn[ii][2], &qtn[ii][3]);
            if(finint!=4) fail_read(procid, qouts, "expected 4 numeric columns per row for all string replicas");
        }
        fclose(in);
    } else {
        tistr();
        printf("\'%s\' not found; using default identity quaternions (procid=%d, Time: %s).\n",qouts,procid,tms);
    }
    
    if(argc>2){
        Py = atof(argv[2]);
        tistr();
        printf("Overwriting input on procid=%d: Setting Py=%lf (Time: %s).\n",procid,Py,tms);
    }
    validate_protein_settings(procid, protein_input_name);
    // D[2] = 2*pi; ///HARDCODED
    
    double V = D[2]*D[1]*D[0];
    phidb = Vc; // interpreted by density::props according to ens
    if(procid==0){
        printf("Input:\n");
        printf("%lf %lf %lf %lf %lf (%lE)\n",chi[0],chi[1],chi[2],f,Vc,phidb);
        printf("%d %d %d %d\n",N,m[0],m[1],m[2]);
        printf("%lf %lf %lf\n",D[0],D[1],D[2]);
        printf("justFE=%d dostring=%d perp_update=%d dangle=%d fix=%d %d ens=%d\n", justFE, dostring, perp_update, dangle, fix[0], fix[1], ens);
        if(justFE==1) printf("WARNING: justFE=1 so the outer times loop is skipped and only the free energy is evaluated from the current fields.\n");
        if(dostring==1) printf("String coupling enabled.\n");
        else printf("String coupling disabled; replicas relax independently.\n");
        if(perp_update==1) printf("Perpendicular string projection enabled when redistribution starts.\n");
        else printf("Perpendicular string projection disabled.\n");
        if(dangle==1) printf("Dangle string mode enabled: first redistribution length is retained and the last replica is redistributed.\n");
        if(ens==1) printf("Using the canonical ensemble interpretation: Vc/phidb is the copolymer volume fraction.\n");
        if(ens==2) printf("Using the GC/semi-grand interpretation: Vc/phidb is the copolymer fugacity.\n");
        printf("%lf %lf %lf\n",Pr,Py, Pth);
        if(protein_have_Pth_file) printf("per-replica Pth values loaded from Pth\n");
        printf("Prx=%lf\n", protein_Prx);
        printf("%lf %lf %lf %lf\n",Px00,Py00, Pz00, Pth00);
        printf("%lf %lf %lf %lf\n",pro1,pro2,pro3,protein_mvin);
        printf("%d %d\n",protein_enabled,protein_movable);
        printf("protein family header: %s\n", PROTEIN_HEADER);
        printf("pitch=%lf hydrophilic_mode=%d patch_wx=%lf patch_wy=%lf patch_offset2=%lf patch_offset3=%lf\n",
               protein_pitch, protein_hydrophilic_mode, protein_patch_wx, protein_patch_wy,
               protein_patch_offset2, protein_patch_offset3);
        printf("protein_symmetrize=%d protein_pivot_align=%d\n", protein_symmetrize, protein_pivot_align);
        printf("snare_n=%d snare_x0=%lf snare_ring_radius=%lf snare_length=%lf snare_radius=%lf snare_pad=%lf\n",
               snare_n, snare_x0, snare_ring_radius, snare_length, snare_radius, snare_pad);
        printf("checkpoint_precision=%d\n",checkpoint_precision);
        printf("%d\n",flag);
#ifdef USE_HDF5
        printf("HDF5 output enabled. Parallel HDF5 build available: %s. Using rank-0 writer.\n", hdf5_parallel_build_available() ? "yes" : "no");
#else
        printf("HDF5 output disabled at build time.\n");
#endif
    }
    
    // Decide which processors own which string replicas.
    // Replicas are assigned in contiguous blocks, as evenly as possible.
    // Any leftover replicas go to the first few ranks. If numprocs>strn, the
    // extra ranks simply get ndo=0 and do no replica work.
    int *doiis, *ndo, *whois;
    
    doiis = new int[strn];
    whois = new int[strn];
    ndo = new int[numprocs];
    
    for(ii=0;ii<strn;ii++) doiis[ii]=0;
    for(ii=0;ii<strn;ii++) whois[ii]=-1;

    int ndo0 = strn/numprocs;
    int rem = strn%numprocs;
    for(ii=0;ii<numprocs;ii++){
        ndo[ii] = ndo0;
        if(ii<rem) ndo[ii]++;
    }

    int start = 0;
    for(int p=0; p<numprocs; p++){
        for(int j=0; j<ndo[p]; j++){
            int iii = start + j;
            whois[iii] = p;
            if(p==procid) doiis[iii] = 1;
        }
        start += ndo[p];
    }

    if(procid==0){
        int ndo_min=ndo[0], ndo_max=ndo[0], ndosum=0;
        for(ii=0;ii<numprocs;ii++){
            ndosum += ndo[ii];
            ndo_min = min(ndo_min, ndo[ii]);
            ndo_max = max(ndo_max, ndo[ii]);
        }
        for(ii=0;ii<strn;ii++){
            if(whois[ii]<0 || whois[ii]>=numprocs){
                printf("Fatal error: invalid owner for ii=%d: %d\n",ii,whois[ii]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        if(ndosum!=strn){
            printf("Fatal error: replica ownership sum mismatch: ndosum=%d strn=%d\n",ndosum,strn);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if(ndo_max-ndo_min>1){
            printf("Fatal error: uneven replica distribution: min=%d max=%d\n",ndo_min,ndo_max);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("Replica ownership:\n");
        for(ii = 0; ii < strn; ii++) printf("%d ", whois[ii]);
        printf("\n");
    }
    
    M = m[0]*m[1]*m[2];
    Mk = m[0]*m[1]*(m[2]/2+1);
    
    NA=(int) round (N*f), NB=N-NA;
    f = double(NA)/N;
    dsA = f/NA, dsB = (1-f)/NB;
    
    density *D2 = new density(N, D);
    splineChunkM = chooseSplineChunkM(M, procid);
    D2->setupSplineBuffers(splineChunkM);
    
    
    malloc3d(&DEV, strn, DIM, 3*M);
    malloc3d(&DDEV, strn, DIM, 3*M);
    malloc3d(&WIN, strn, DIM, 3*M);
    
    malloc2d(&prott1,strn,M);
    malloc2d(&prott2,strn,M);
    malloc2d(&prott3,strn,M);
    
    malloc2d(&dxprott1,strn,M);
    malloc2d(&dxprott2,strn,M);
    malloc2d(&dxprott3,strn,M);
    malloc2d(&dyprott1,strn,M);
    malloc2d(&dyprott2,strn,M);
    malloc2d(&dyprott3,strn,M);
    malloc2d(&dzprott1,strn,M);
    malloc2d(&dzprott2,strn,M);
    malloc2d(&dzprott3,strn,M);
    malloc2d(&dFdR,strn,3);
    
    
    malloc2d(&Fxfield,strn,M);
    malloc2d(&Fyfield,strn,M);
    malloc2d(&Fzfield,strn,M);
    
    malloc2d(&forces,strn,3);
    malloc2d(&tors,strn,3);
    //tors = (double *) malloc(strn*sizeof(double ));
    
    malloc3d(&q1, strn, N+1, M);
    malloc3d(&q2, strn, N+1, M);
    
    // set lookup table for laplacian operator
    expKA = (double *) malloc(Mk*sizeof(double )); //new double[Mk];
    expKB = (double *) malloc(Mk*sizeof(double )); // new double[Mk];
    expKA2 = (double *) malloc(Mk*sizeof(double )); // new double[Mk]; // for simpson method
    expKB2 = (double *) malloc(Mk*sizeof(double )); // new double[Mk];
    
    
    // `W[ii]` stores both primary SCFT fields and derived work arrays:
    //   [0:M)     = field 1
    //   [M:2*M)   = field 2
    //   [2*M:3*M) = pressure / incompressibility-like field
    //   [3*M:11*M)= densities and helper fields
    //   [11*M:16*M)= string-method work buffers
    malloc2d(&W,strn,16*M);
    phiA = (double **)malloc(strn * sizeof(double *));
    phiB = (double **)malloc(strn * sizeof(double *));
    phih = (double **)malloc(strn * sizeof(double *));
    phicB = (double **)malloc(strn * sizeof(double *));
    Wp = (double **)malloc(strn * sizeof(double *));
    Wm = (double **)malloc(strn * sizeof(double *));
    phip = (double **)malloc(strn * sizeof(double *));
    phim = (double **)malloc(strn * sizeof(double *));
    
    dWda = (double **)malloc(strn * sizeof(double *));
    //dSCFTW = (double **)malloc(strn * sizeof(double *));
    //Wnew = (double **)malloc(strn * sizeof(double *));
    dSCFTWN = (double **)malloc(strn * sizeof(double *));
    
    for(ii=0;ii<strn;ii++){
        phiA[ii]=W[ii]+3*M;
        phiB[ii]=W[ii]+4*M;
        phih[ii]=W[ii]+5*M;
        phicB[ii]=W[ii]+6*M;
        Wp[ii]= W[ii]+7*M;
        Wm[ii]= W[ii]+8*M;
        phip[ii]= W[ii]+9*M;
        phim[ii]= W[ii]+10*M;
        dWda[ii]=W[ii]+11*M; // string derivative buffer for the two composition-like fields
        dSCFTWN[ii]=W[ii]+13*M;
        
    }
    
    for(ii=0;ii<strn;ii++){
        mkprot(procid,chi,qtn[ii],doiis,ii);
        difprot(procid,ii);
    }
    
    if ((flag==1 || flag==2) && fexist("wins.h5")) {
#ifdef USE_HDF5
        int win_fields = read_wins_hdf5("wins.h5", W, m);
        if(win_fields < 0) fail_read(procid, "wins.h5", "failed to read HDF5 restart fields");
        tistr();
        if(procid==0) printf("Read wins.h5 with %d field dataset(s) (Time: %s).\n", win_fields, tms);
        for(ii=0;ii<strn-1;ii++) if(EDist(W,ii,ii+1)<1E-4) {
            printf("matches(%d): %d,%d\n",procid,ii,ii+1);
            springsteps += 1;
        }
#else
        fail_read(procid, "wins.h5", "HDF5 restart file found but this executable was built without HDF5 support");
#endif
    } else if (flag==1) {
        for(ii=0;ii<strn;ii++){ // CHANGE BACK TO strn
            sprintf(winf,"win%d",ii);
            int win_lines = lines(winf);
            int win_cols = file_columns(winf);
            if(!fexist(winf)) fail_read(procid, winf, "required win file not found for flag==1");
            if(win_lines!=m[0]*m[1]*m[2]){
                char detail[200];
                sprintf(detail, "expected %d data lines but found %d", m[0]*m[1]*m[2], win_lines);
                fail_read(procid, winf, detail);
            }
            if(!(win_cols==2 || win_cols==3)){
                char detail[200];
                sprintf(detail, "expected 2 or 3 numeric columns but found %d", win_cols);
                fail_read(procid, winf, detail);
            }
            in=fopen(winf,"r");
            if(in==NULL) fail_read(procid, winf, "failed to open win file");
                if(win_cols==3){
                    for (r=0;r<M;r++) {
                        finint=fscanf(in,"%lf %lf %lf",&W[ii][r],&W[ii][r+M],&W[ii][r+2*M]);
                        if(finint!=3) fail_read(procid, winf, "expected 3 numeric columns in every data row");
                    }
                } else {
                    double w1, wp;
                    for (r=0;r<M;r++) {
                        finint=fscanf(in,"%lf %lf",&w1,&wp);
                        if(finint!=2) fail_read(procid, winf, "expected 2 numeric columns in every data row");
                        W[ii][r] = w1;
                        W[ii][r+M] = w1;
                        W[ii][r+2*M] = wp;
                    }
                }
                fclose(in);
                tistr();
                if(doiis[ii]==1) printf("win%d read by procid=%d: %d %d %d grid, %d lines, %d columns (Time: %s).\n",ii,procid,m[0],m[1],m[2],m[0]*m[1]*m[2],win_cols,tms);
            
            if(numprocs>1) MPI_Barrier(MPI_COMM_WORLD);
        }
        for(ii=0;ii<strn-1;ii++) if(EDist(W,ii,ii+1)<1E-4) {
            printf("matches(%d): %d,%d\n",procid,ii,ii+1);
            springsteps += 1;
        }
        
    } else if (flag==2) {
        for(ii=0;ii<strn;ii++){ // CHANGE BACK TO strn
            sprintf(winf,"win%d",ii);
            if(!fexist(winf)) fail_read(procid, winf, "required win file not found for flag==2");
            {
                int win_lines = lines(winf);
                int win_cols = file_columns(winf);
                if(win_lines!=m[0]*m[1]*m[2]){
                    char detail[200];
                    sprintf(detail, "expected %d data lines but found %d", m[0]*m[1]*m[2], win_lines);
                    fail_read(procid, winf, detail);
                }
                if(win_cols!=2){
                    char detail[200];
                    sprintf(detail, "expected exactly 2 numeric columns but found %d", win_cols);
                    fail_read(procid, winf, detail);
                }
            }
            in=fopen(winf,"r");
            if(in==NULL) fail_read(procid, winf, "failed to open win file");
            for (r=0;r<M;r++) {
                finint=fscanf(in,"%lf %lf",&W[ii][r],&W[ii][r+2*M]);
                if(finint!=2) fail_read(procid, winf, "expected 2 numeric columns in every data row");
            }
            fclose(in);
            for (r=0;r<M;r++) W[ii][r+M] = W[ii][r];
            tistr();
            printf("win%d read by procid=%d (Time: %s).\n",ii,procid,tms);
        }
        
        
    } else {
        for (x=0; x<m[0]; x++)
            for (y=0; y<m[1]; y++)
                for (z=0; z<m[2]; z++) { //initial W here - right now parallel to walls
                    r = (x*m[1]+y)*m[2]+z;
                    for(ii=0;ii<strn;ii++){
                        W[ii][r] = chi[0];
                        W[ii][r+M] = chi[0];
                        W[ii][r+2*M] = -2.0;
                    }
                    ///3Ds:
                    double rr2 = (z-m[2]/2.0)*(z-m[2]/2.0) + (y-m[1]/2.0)*(y-m[1]/2.0);
                    double rr = sqrt(rr2);
                    
                    double r0 = m[1]/4.0 - 6;
                    // Keep these coarse initialization families easy to toggle locally.
                    if(1==1){ // Keep this explicit branch as the default initialization pattern.
                        W[0][r]  += -1.0*chi[0]*exp(-1.0*(pow((rr-r0)/8.0,2.0)));
                        W[0][r+M]+= -1.0*chi[1]*exp(-1.0*(pow((rr-r0)/8.0,2.0)));
                    }
                }
    }
    
    
    FEs0   = (double *) malloc(strn*sizeof(double ));
    FEs1   = (double *) malloc(strn*sizeof(double ));
    FEs2   = (double *) malloc(strn*sizeof(double ));
    FEs    = (double *) malloc(strn*sizeof(double ));
    phctot = (double *) malloc(strn*sizeof(double ));
    
    tistr();
    printf("Setup (%d) (Time: %s)\n",procid,tms);
    
    int nsterp=field_iterations; //2E3
    double alf[strn];
    ii=0;
    
    ii=strn-1;
    
    for(ii=0;ii<strn;ii++){
        mkprot(procid,chi,qtn[ii],doiis,ii);
        difprot(procid,ii);
    }
    tistr();
    printf("Prots setup (%d) (Time: %s)\n",procid,tms);

    for(ii=0;ii<strn;ii++) {
        if(doiis[ii]==1) {
            solve_field0(W, chi, f, D2, ii,1E2, 1E-4,0);
#ifndef USE_HDF5
            sprintf(outfl,"win%d",ii);
            out=fopen(outfl,"w");
            for (r=0;r<M;r++) fprintf(out,"%.6lf %.6lf %.6lf\n",W[ii][r],W[ii][r+M],W[ii][r+2*M]);
            fclose(out);
#endif
        }
        
    }

#ifdef USE_HDF5
    write_hdf5_outputs(W, D2, f, procid);
#endif

    tistr();
    printf("Here1 (%d) (Time: %s)\n",procid,tms);
    
    double sforce[3];
    double move[strn+1][4],movesp[strn+1][4];//moveav[4]
    double mfrac=10.0; //speed of grad descent (proportionality between amount moved and force)
    double mfract=20.0; //same but for torques
    double maxmove=0.03; //maximum amount that prot moves
    double maxang=0.0*2*pi/180;
    int MAXtime=max_outer;
    sforce[0]=0; sforce[1]=0; sforce[2]=0;
    int dosp=1;
    double eps=0.2;
    sprintf(outfl,"Forces0");
    out = fopen(outfl,"w");
    fclose(out);
    sprintf(outfl,"Forces");
    out = fopen(outfl,"w");
    fclose(out);
    
    tistr();
    printf("Here2 (%d) (Time: %s)\n",procid,tms);

    int ank=0;
    if(justFE!=1)
    for(int times=0;times<MAXtime;times++){
        //place protein and calculate dw/dr
        for(ii=0;ii<strn;ii++){
            mkprot(procid,chi,qtn[ii],doiis,ii,times);
            difprot(procid,ii);
        }
        ank = solve_field(W,chi,f,D2,alf,doiis,ndo,whois,procid,numprocs,dostring,-1,nsterp);
        for(ii=0;ii<strn;ii++) if(doiis[ii]==1) solve_field0(W, chi, f, D2, ii, 2E2, 1E-4);

        springsteps=0; //reset springsteps assuming whatever issue is taken care of
        
        // PB
        FE=FreeE(W, chi, f, D2, alf, doiis, ndo, whois, procid, numprocs, 0);
        if(procid==0){
            sprintf(outfl,"FEs_move");
            out = fopen(outfl,"w");
            for(ii=0;ii<strn;ii++) fprintf(out,"%lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\n",alf[ii],FEs[ii],FEs0[ii],FEs1[ii],phctot[ii]);
            fclose(out);
                        sprintf(outfl,"FEs");
                        out = fopen(outfl,"w");
                        for(ii=0;ii<strn;ii++) fprintf(out,"%lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\n",alf[ii],FEs[ii],FEs0[ii],FEs1[ii],FEs2[ii],phctot[ii]);
                        fclose(out);
}
        //calculate forces
        Fpartfield(doiis, D2, f);
        for(ii=0;ii<strn;ii++) {
            if(doiis[ii]==1) {
                torque(ii,tors[ii]);
                sumforce(forces,ii);
            }
            }
        //print forces
        sprintf(outfl,"Forces");
        out = fopen(outfl,"a");
        fprintf(out,"%lf ",Pz0[0]);
        for(ii=0;ii<strn;ii++) {
            for(int jj=0;jj<3;jj++)
                fprintf(out,"%lf ",forces[ii][jj]);
        }
        for(ii=0;ii<strn;ii++) {
            for(int jj=0;jj<3;jj++)
                fprintf(out,"%lf ",dFdR[ii][jj]);
        }
        for(ii=0;ii<strn;ii++)
            fprintf(out,"%lf ",FEs[ii]);
        fprintf(out,"\n");
        fclose(out);
        //
        
        
        //sum forces over ii
        if(numprocs>1){
            //for(ii=0;ii<strn;ii++) MPI_Bcast(&tors[ii], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
            for(int jj=0;jj<3;jj++){
                for(ii=0;ii<strn;ii++) MPI_Bcast(&forces[ii][jj], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
                for(ii=0;ii<strn;ii++) MPI_Bcast(&tors[ii][jj], 1, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
            }
        }

        if(fix[0]==1){
            for(int jj=0;jj<3;jj++) forces[0][jj]=tors[0][jj]=0.0;
        }
        if(fix[1]==1){
            for(int jj=0;jj<3;jj++) forces[strn-1][jj]=tors[strn-1][jj]=0.0;
        }
        
        //average force stuff
        sforce[0]=0; sforce[1]=0; sforce[2]=0;
        for(ii=0;ii<strn;ii++){
            for(int jj=0;jj<3;jj++) sforce[jj]+=forces[ii][jj]/(1.0*strn);
        }
        
        
        move[strn][0]=0;
        move[strn][1]=0;
        move[strn][2]=0;
        move[strn][3]=0;
        
        //individual forces
        for(ii=0;ii<strn;ii++){
            for(int jj=0;jj<3;jj++) move[ii][jj]=cap(forces[ii][jj]/mfrac,maxmove);
            move[ii][3]=0;
        }
        
        //attract particle to.. self.. over time...
        // smoothes out trajectory
        if(dosp==1)
            for(ii=0;ii<strn;ii++){
                if(ii==0){
                    movesp[ii][0] = 0.0*cap(eps*(Px0[ii+1]-Px0[ii]),maxmove);
                    movesp[ii][1] = 0.0*cap(eps*(Py0[ii+1]-Py0[ii]),maxmove);
                    movesp[ii][2] = 0.0*cap(eps*(Pz0[ii+1]-Pz0[ii]),maxmove);
                } else if(ii==(strn-1)){
                    movesp[ii][0] = 0.5*cap(eps*(Px0[ii-1]-Px0[ii]),maxmove);
                    movesp[ii][1] = 0.5*cap(eps*(Py0[ii-1]-Py0[ii]),maxmove);
                    movesp[ii][2] = 0.5*cap(eps*(Pz0[ii-1]-Pz0[ii]),maxmove);
                } else{
                    movesp[ii][0] = cap(eps*(Px0[ii-1]+Px0[ii+1] - 2.0*Px0[ii]),maxmove);
                    movesp[ii][1] = cap(eps*(Py0[ii-1]+Py0[ii+1] - 2.0*Py0[ii]),maxmove);
                    movesp[ii][2] = cap(eps*(Pz0[ii-1]+Pz0[ii+1] - 2.0*Pz0[ii]),maxmove);
                }
                
            }
        //
        
        //update positions
        if(protein_movable && protein_enabled && ank>40)
            for(ii=0;ii<strn;ii++){ //1 to strn-1 to not update last point
                if((fix[0]==1 && ii==0) || (fix[1]==1 && ii==(strn-1))) continue;
                Px0[ii] += move[ii][0] + movesp[ii][0];
                Py0[ii] += move[ii][1] + movesp[ii][1];
                Pz0[ii] += move[ii][2] + movesp[ii][2];
                
                //rotate:
                qinit(dqtnx[ii],-cap(tors[ii][0]/mfract,maxang),1);
                qinit(dqtny[ii],-cap(tors[ii][1]/mfract,maxang),2);
                qinit(dqtnz[ii],-cap(tors[ii][2]/mfract,maxang),3);
                
                //adjust qtn appropriately
                qmult(qtn[ii], dqtnx[ii], qtn[ii]);
                qmult(qtn[ii], dqtny[ii], qtn[ii]);
                qmult(qtn[ii], dqtnz[ii], qtn[ii]);
                normq(qtn[ii]);
            }
        tistr();
        if(procid==0) printf("times=%d force = %lf %lf %lf, motion=%lf %lf %lf. position=%lf %lf. FE0=%.9lg (Time: %s)\n",times,sforce[0],sforce[1],sforce[2],move[0][0],move[0][1],move[0][2],Py0[0],Pz0[0],FEs[0],tms);
        
        if(procid==0){
            write_keyed_output("output", chi, f, Vc, N, m, D, 1);
            write_keyed_protein_input(protein_input_name, Pr, Py, Pth, Px00, Py00, Pz00, Pth00, pro1, pro2, pro3, protein_mvin, protein_enabled, protein_movable);
        }
        
        if(procid==0){
            out = fopen(pouts,"w"); //P0s
            for(ii=0;ii<strn;ii++) fprintf(out,"%lf %lf %lf %lf\n",Px0[ii],Py0[ii], Pz0[ii], Pth0[ii]);
            fclose(out);
            
            out = fopen(qouts,"w"); //qtns
            for(ii=0;ii<strn;ii++) fprintf(out,"%lf %lf %lf %lf\n",qtn[ii][0],qtn[ii][1], qtn[ii][2], qtn[ii][3]);
            fclose(out);
            
            ensure_dir_exists("moves");
            
            out = fopen("moves/moves","w");
            for(ii=0;ii<strn;ii++) fprintf(out,"%lf %lf %lf %lf\n",move[ii][0],move[ii][1], move[ii][2], move[ii][3]);
            fclose(out);
            
            out = fopen("moves/moves0","w");
            for(ii=0;ii<strn;ii++)  fprintf(out,"%lf %lf %lf %lf\n",forces[ii][0]/mfrac,forces[ii][1]/mfrac, forces[ii][2]/mfrac, forces[ii][3]/mfrac);
            fclose(out);
            
            sprintf(outfl,"moves/movesX"); out = fopen(outfl,"a");
            for(ii=0;ii<strn;ii++) fprintf(out,"%lf ",move[ii][0]);
            fprintf(out,"\n"); fclose(out);
            
            sprintf(outfl,"moves/movesY"); out = fopen(outfl,"a");
            for(ii=0;ii<strn;ii++) fprintf(out,"%lf ",move[ii][1]);
            fprintf(out,"\n"); fclose(out);
            
            sprintf(outfl,"moves/movesZ"); out = fopen(outfl,"a");
            for(ii=0;ii<strn;ii++) fprintf(out,"%lf ",move[ii][2]);
            fprintf(out,"\n"); fclose(out);
            
            sprintf(outfl,"moves/movesspX"); out = fopen(outfl,"a");
            for(ii=0;ii<strn;ii++) fprintf(out,"%lf ",movesp[ii][0]);
            fprintf(out,"\n"); fclose(out);
            
            sprintf(outfl,"moves/movesspY"); out = fopen(outfl,"a");
            for(ii=0;ii<strn;ii++) fprintf(out,"%lf ",movesp[ii][1]);
            fprintf(out,"\n"); fclose(out);
            
            sprintf(outfl,"moves/movesspZ"); out = fopen(outfl,"a");
            for(ii=0;ii<strn;ii++) fprintf(out,"%lf ",movesp[ii][2]);
            fprintf(out,"\n"); fclose(out);
            
            
            sprintf(outfl,"moves/P0sX"); out = fopen(outfl,"a");
            for(ii=0;ii<strn;ii++) fprintf(out,"%lf ",Px0[ii]);
            fprintf(out,"\n"); fclose(out);
            
            sprintf(outfl,"moves/P0sY");
            out = fopen(outfl,"a");
            for(ii=0;ii<strn;ii++) fprintf(out,"%lf ",Py0[ii]);
            fprintf(out,"\n"); fclose(out);
            
            sprintf(outfl,"moves/P0sZ"); out = fopen(outfl,"a");
            for(ii=0;ii<strn;ii++) fprintf(out,"%lf ",Pz0[ii]);
            fprintf(out,"\n"); fclose(out);
            
            
        }
        
        if(sigg==7) break;
    }
    
    
    double Ri=4, Rf=0;
    double DR = (Rf-Ri)/(strn-1);//,theta, dxxx,dtheta,s2=sqrt(2.0);
    
    
    if(numprocs>1){
        for(ii=0;ii<strn;ii++)
            MPI_Bcast(W[ii], M*3, MPI_DOUBLE, whois[ii], MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    tistr();
    printf("Mixed (%d) (Time: %s)\n",procid,tms);
    
    FE=0;
    
    FE=FreeE(W, chi, f, D2, alf, doiis, ndo, whois, procid, numprocs, 0);
    tistr();
    printf("Free E calculated (%d) (Time: %s)\n",procid,tms);
    
    
    
    printf("%9.5f %9.5f\n",chi[0],FE);
    tistr();
    printf("FE calculated (%d) (Time: %s)\n",procid,tms);
    
    
    if(FE==FE && procid==0){ //only output field if not nan
        sprintf(outfl,"FEs");
        out = fopen(outfl,"w");
        for(ii=0;ii<strn;ii++) fprintf(out,"%lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\n",alf[ii],FEs[ii],FEs0[ii],FEs1[ii],phctot[ii]);
        fclose(out);
        
        write_keyed_output("output", chi, f, Vc, N, m, D, 1);
        write_keyed_protein_input(protein_input_name, Pr, Py, Pth, Px00, Py00, Pz00, Pth00, pro1, pro2, pro3, protein_mvin, protein_enabled, protein_movable);
        
        out = fopen(pouts,"w");
        for(ii=0;ii<strn;ii++) {
            fprintf(out,"%lf %lf %lf %lf\n",Px0[ii],Py0[ii], Pz0[ii], Pth0[ii]);
        }
        fclose(out);
        
        tistr();
        printf("output files printed (%d) (Time: %s)\n",procid,tms);
#ifdef USE_HDF5
        write_hdf5_outputs(W, D2, f, procid);
#else
        for(ii=0;ii<strn;ii++){
            sprintf(outfl,"win%d",ii);
            out=fopen(outfl,"w");
            for (r=0;r<M;r++) fprintf(out,"%.6lf %.6lf %.6lf\n",W[ii][r],W[ii][r+M],W[ii][r+2*M]);
            fclose(out);
        }
        
        for(ii=0;ii<strn;ii++)
            for(r=0;r<M;r++){
                Wm[ii][r]=W[ii][r]-W[ii][r+M];
                Wp[ii][r]=W[ii][r]+W[ii][r+M];
                phip[ii][r]=phiA[ii][r]+phiB[ii][r];
            }

        for(ii=0;ii<strn;ii++){
            if(doiis[ii]==1){
                sprintf(outfl,"rhoA_%d.vtk",ii);
                tovtk(outfl, m, D, phiA[ii]);
            }
        }
#endif
    }
    
    
    tistr();
    printf("view / restart fields printed (%d) (Time: %s)\n",procid,tms);
    
    
    
    tistr();
    printf("Done (%d) (Time: %s)\n",procid,tms);
    
    
}
