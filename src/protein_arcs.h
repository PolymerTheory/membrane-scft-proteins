#pragma once

// This header intentionally defines `mkprot()` so different protein geometries
// can be swapped by changing a single include in `scft.cu`.
// This implementation covers one wrapped-protein family:
// - ring: full angular coverage, zero pitch
// - arc: partial angular coverage, zero pitch
// - helix: nonzero pitch
// Optional controls add:
// - hydrophilic mode `surface` (current default) or `patch`
// - angular patch offsets for `prott2` and `prott3`
// - mirrored duplication (`protein_symmetrize`)
// - optional midpoint-preserving alignment shift (`protein_pivot_align`)

// Canonical arcs: 0=uncapped, 1=aligned generic caps.
// Reflection: 0=single, 1=additive pair, 2=legacy half-domain copy (comparison).
static inline void arcs_reflect(int ii)
{
    if(!protein_symmetrize) return;
    for(int x=0;x<m[0];x++) for(int y=0;y<m[1];y++)
        for(int z=0;z<=m[2]/2;z++){
            const int zp=(m[2]-z)%m[2];
            const int r=(x*m[1]+y)*m[2]+z;
            const int rp=(x*m[1]+y)*m[2]+zp;
            if(protein_symmetrize==1){
                // Read both original values before writing either side.
                const double a=prott1[ii][r]+prott1[ii][rp];
                const double b=prott2[ii][r]+prott2[ii][rp];
                const double c=prott3[ii][r]+prott3[ii][rp];
                prott1[ii][r]=prott1[ii][rp]=a;
                prott2[ii][r]=prott2[ii][rp]=b;
                prott3[ii][r]=prott3[ii][rp]=c;
            }else if(z>0 && z<m[2]/2.0){
                prott1[ii][rp]=prott1[ii][r];
                prott2[ii][rp]=prott2[ii][r];
                prott3[ii][rp]=prott3[ii][r];
            }
        }
}

static inline double protein_wrap_angle(double phi)
{
    while(phi < 0.0) phi += 2.0*pi;
    while(phi >= 2.0*pi) phi -= 2.0*pi;
    return phi;
}

static inline int protein_is_full_ring(double phi_extent)
{
    return fabs(phi_extent) >= (2.0*pi - 1.0e-8);
}

static inline double protein_centerline_x(double phi, double phi0, double phi_extent, double pitch)
{
    if(fabs(pitch) < 1.0e-12) return 0.0;
    return pitch*(phi - (phi0 + 0.5*phi_extent))/(2.0*pi);
}

static inline double protein_choose_phi(double th, double xx_local, double phi0, double phi_extent, double pitch)
{
    if(fabs(pitch) < 1.0e-12) return th;

    const int turns = (int)ceil(fabs(phi_extent)/(2.0*pi)) + 3;
    double phi_best = th;
    double err_best = 1.0e300;
    for(int kk=-turns; kk<=turns; kk++){
        double phi_try = th + 2.0*pi*kk;
        double x_try = protein_centerline_x(phi_try, phi0, phi_extent, pitch);
        double err = fabs(xx_local - x_try);
        if(err < err_best){
            err_best = err;
            phi_best = phi_try;
        }
    }
    return phi_best;
}

static inline void protein_zero_fields_for_replica(int ii)
{
    for(int r=0; r<M; r++){
        prott1[ii][r] = 0.0;
        prott2[ii][r] = 0.0;
        prott3[ii][r] = 0.0;
    }
}

static inline double protein_cap_body_distance(double xx_local, double dy_local, double wx, double wy, double nn)
{
    return pow(fabs(xx_local)/wx, nn) + pow(fabs(dy_local)/wy, nn);
}

static inline double protein_cap_point_distance(double xx, double yy, double zz,
                                                double xc, double yc, double zc,
                                                double wx, double wy, double wz)
{
    double dx = (xx-xc)/wx;
    double dy = (yy-yc)/wy;
    double dz = (zz-zc)/wz;
    return dx*dx + dy*dy + dz*dz;
}

static inline void protein_patch_center(double x_center, double y_center, double z_center,
                                        double offset, double radial_shift,
                                        double y_ref, double z_ref,
                                        double &xc, double &yc, double &zc)
{
    double er_y = 1.0;
    double er_z = 0.0;
    double rr = sqrt(y_ref*y_ref + z_ref*z_ref);
    if(rr > 1.0e-12){
        er_y = y_ref/rr;
        er_z = z_ref/rr;
    }
    double dx_local = -radial_shift*sin(offset);
    double dy_local =  radial_shift*cos(offset);
    xc = x_center + dx_local;
    yc = y_center - dy_local*er_y;
    zc = z_center - dy_local*er_z;
}

static inline void protein_apply_output(int procid, int ii, int times)
{
    char outfl[100];
    if(procid==0 && times>-1 && ii==(strn-1)){
#ifdef USE_HDF5
        if(write_proteins_hdf5_atomic("proteins.h5", m, prott1, prott2, prott3)){
            write_proteins_xdmf("proteins.xdmf", "proteins.h5", m, D);
        }
#else
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
#endif
    }
}

void mkprot (int procid, double *chi, double *qts, int *doiis, int ii, int times=-1)
{
    const double nn = 2.0;
    const double Prx = protein_Prx;
    const double patch_wx = protein_patch_wx;
    const double patch_wy = protein_patch_wy;
    const double mvin = protein_mvin;
    const double phi0 = Pth0[ii];
    const double phi1 = Pth0[ii] + Pth;
    const double pitch = protein_pitch;
    const int full_ring = protein_is_full_ring(Pth) && fabs(pitch) < 1.0e-12;
    const int patch_mode = (protein_hydrophilic_mode==1);
    const double alpha2 = protein_patch_offset2;
    const double alpha3 = protein_patch_offset3;
    const double pivot_ang = 2.0*atan2(qts[2], qts[0]);
    const double midpoint_z = Py*sin(phi0 + 0.5*Pth);
    // Keep the arc midpoint at the same lab-frame point as q=identity.
    const double pivot_dx = protein_pivot_align ? midpoint_z*sin(pivot_ang) : 0.0;
    const double pivot_dz = protein_pivot_align ? midpoint_z*(1.0-cos(pivot_ang)) : 0.0;
    const double helix_span = protein_centerline_x(phi1, phi0, Pth, pitch) - protein_centerline_x(phi0, phi0, Pth, pitch);

    protein_zero_fields_for_replica(ii);
    if(!protein_enabled){
        protein_apply_output(procid, ii, times);
        return;
    }

    const double x_end0 = -0.5*helix_span;
    const double x_end1 =  0.5*helix_span;
    const double y_end0 = Py*cos(phi0);
    const double z_end0 = Py*sin(phi0);
    const double y_end1 = Py*cos(phi1);
    const double z_end1 = Py*sin(phi1);
    const double y_end0_surface2 = Py*cos(phi0);
    const double z_end0_surface2 = Py*sin(phi0);
    const double y_end1_surface2 = Py*cos(phi1);
    const double z_end1_surface2 = Py*sin(phi1);
    double cap2_x0, cap2_y0, cap2_z0, cap2_x1, cap2_y1, cap2_z1;
    double cap3_x0, cap3_y0, cap3_z0, cap3_x1, cap3_y1, cap3_z1;

    protein_patch_center(x_end0, y_end0, z_end0, alpha3, mvin, y_end0, z_end0, cap3_x0, cap3_y0, cap3_z0);
    protein_patch_center(x_end1, y_end1, z_end1, alpha3, mvin, y_end1, z_end1, cap3_x1, cap3_y1, cap3_z1);

    if(patch_mode){
        protein_patch_center(x_end0, y_end0, z_end0, alpha2, mvin, y_end0, z_end0, cap2_x0, cap2_y0, cap2_z0);
        protein_patch_center(x_end1, y_end1, z_end1, alpha2, mvin, y_end1, z_end1, cap2_x1, cap2_y1, cap2_z1);
    } else {
        cap2_x0 = x_end0; cap2_y0 = y_end0_surface2; cap2_z0 = z_end0_surface2;
        cap2_x1 = x_end1; cap2_y1 = y_end1_surface2; cap2_z1 = z_end1_surface2;
    }

    double vv[4]; vv[0]=0.0;
    for (int x=0; x<m[0]; x++)
        for (int y=0; y<m[1]; y++)
            for (int z=0; z<m[2]; z++) {
                int r = (x*m[1]+y)*m[2]+z;

                double zz = z*D[2]/m[2];
                double yy = y*D[1]/m[1];
                double xx = x*D[0]/m[0];

                double zz2 = zz-(D[2]/2.0 + Pz0[ii] + pivot_dz);
                double yy2 = yy-(D[1]/2.0 + Py0[ii]);
                double xx2 = xx-(D[0]/2.0 + Px0[ii] + pivot_dx);

                vv[1]=xx2; vv[2]=yy2; vv[3]=zz2;
                rotvecq(qts, vv, vv);
                xx2=vv[1]; yy2=vv[2]; zz2=vv[3];

                double rr = sqrt(zz2*zz2 + yy2*yy2);
                double th = protein_wrap_angle(atan2(zz2,yy2));
                double phi_near = full_ring ? th : protein_choose_phi(th, xx2, phi0, Pth, pitch);
                double x_center = protein_centerline_x(phi_near, phi0, Pth, pitch);
                double x_local = xx2 - x_center;
                double dy_local = Py - rr;

                double dr1_body = protein_cap_body_distance(x_local, dy_local, Prx, Pr, nn);

                double dr2_body, dr3_body;
                double x3 = x_local*cos(alpha3) + dy_local*sin(alpha3);
                double y3 = -x_local*sin(alpha3) + dy_local*cos(alpha3);
                if(patch_mode){
                    double x2 = x_local*cos(alpha2) + dy_local*sin(alpha2);
                    double y2 = -x_local*sin(alpha2) + dy_local*cos(alpha2);
                    dr2_body = pow(fabs(x2)/patch_wx, nn) + pow(fabs(y2-mvin)/patch_wy, nn);
                    dr3_body = pow(x3/patch_wx, 2.0) + pow((y3-mvin)/patch_wy, 2.0);
                } else {
                    dr2_body = protein_cap_body_distance(x_local, dy_local, Prx, Pr, nn);
                    dr3_body = pow(x3/patch_wx, 2.0) + pow((y3-mvin)/patch_wy, 2.0);
                }

                int in_body = full_ring || (phi_near>=phi0 && phi_near<=phi1);
                if(in_body){
                    prott1[ii][r] = pro1*chi[0]*exp(-dr1_body);
                    prott2[ii][r] = pro2*chi[0]*exp(-dr2_body);
                    prott3[ii][r] = pro3*chi[0]*exp(-dr3_body);
                } else {
                    if(protein_cap_mode==0) continue; // all three exterior cap fields stay zero
                    double dr1_cap0 = protein_cap_point_distance(xx2, yy2, zz2, x_end0, y_end0, z_end0, Prx, Pr, Pr);
                    double dr1_cap1 = protein_cap_point_distance(xx2, yy2, zz2, x_end1, y_end1, z_end1, Prx, Pr, Pr);
                    double dr1_cap = (dr1_cap0 < dr1_cap1) ? dr1_cap0 : dr1_cap1;

                    double dr2_cap0, dr2_cap1, dr3_cap0, dr3_cap1;
                    if(patch_mode){
                        dr2_cap0 = protein_cap_point_distance(xx2, yy2, zz2, cap2_x0, cap2_y0, cap2_z0, patch_wx, patch_wy, patch_wy);
                        dr2_cap1 = protein_cap_point_distance(xx2, yy2, zz2, cap2_x1, cap2_y1, cap2_z1, patch_wx, patch_wy, patch_wy);
                    } else {
                        dr2_cap0 = protein_cap_point_distance(xx2, yy2, zz2, cap2_x0, cap2_y0, cap2_z0, Prx, Pr, Pr);
                        dr2_cap1 = protein_cap_point_distance(xx2, yy2, zz2, cap2_x1, cap2_y1, cap2_z1, Prx, Pr, Pr);
                    }
                    dr3_cap0 = protein_cap_point_distance(xx2, yy2, zz2, cap3_x0, cap3_y0, cap3_z0, patch_wx, patch_wy, patch_wy);
                    dr3_cap1 = protein_cap_point_distance(xx2, yy2, zz2, cap3_x1, cap3_y1, cap3_z1, patch_wx, patch_wy, patch_wy);

                    prott1[ii][r] = pro1*chi[0]*exp(-dr1_cap);
                    prott2[ii][r] = pro2*chi[0]*exp(-(dr2_cap0 < dr2_cap1 ? dr2_cap0 : dr2_cap1));
                    prott3[ii][r] = pro3*chi[0]*exp(-(dr3_cap0 < dr3_cap1 ? dr3_cap0 : dr3_cap1));
                }
            }

    arcs_reflect(ii);
    protein_apply_output(procid, ii, times);
}
