// Optional private same-event classical LL/BMT source observer.
// No public/default orbit or tangent path includes this header.
// Reproducible owner: local_secondary_source.native_secondary_header().
#ifndef XTRACK_CAVITY_NATIVE_SECONDARY_H
#define XTRACK_CAVITY_NATIVE_SECONDARY_H
#include "xtrack/beam_elements/elements_src/cavity_native_source.h"

// q=(X,Y,T,Px,Py,E), all particle variables remain runtime inputs.
// profile[0..5]=Re, [6..11]=Im spatial derivatives wrt normalized Z.
// Profiles and raw E/B use the declared positive SOURCE mass normalization,
// uncharged. Q=actual signed charge_e*source_mass_eV/actual_mass_eV ONCE.
// rho=tau0_actual*c*k>=0 is separate; it is NOT inferred from Q*Q.
// anomaly is the actual particle's (g-2)/2.
// No law for photon noise is supplied.
//
// Output rows:
//  0..5 canonical full LL reaction source PER Z (first three exactly zero),
//  6..8 Lorentz BMT omega PER T, 9..11 reacted Thomas extra omega PER T,
// 12..14 raw uncharged E, 15..17 raw uncharged B,
// 18 external power, 19 mechanical LL reaction power, 20 Lorentz emission,
// 21 Schott ENERGY, 22 Lorentz remainder, 23 actual-LL-curve emission,
// 24 corresponding actual remainder. All powers/remainders are PER T.
// Spatial consumers multiply lab-time rates by gamma/uz EXACTLY ONCE;
// reaction_source already contains that factor; Schott is not a rate.
// Higher-rho emission/remainder terms are algebraic consequences of retained
// LL, not higher physical accuracy. Emission is never particle energy loss.
// This observer mutates no particle. Every output is untouched on failure.
GPUFUN
int CavityNative_secondary(const double q[6], double Q, double rho,
        double anomaly, const double profile[12], double aperture,
        double output[25]) {
    if (!isfinite(rho) || rho < 0. || !isfinite(anomaly))
        return CAVITY_NATIVE_ARGUMENT;
    for (int i=0; i<12; ++i)
        if (!isfinite(profile[i])) return CAVITY_NATIVE_NONFINITE;
    double orbit_profile[10], flow[6];
    for (int i=0; i<5; ++i) {
        orbit_profile[i] = profile[i];
        orbit_profile[i+5] = profile[i+6];
    }
    int status = CavityNative_source(q, Q, orbit_profile, aperture, flow);
    if (status != CAVITY_NATIVE_OK) return status;
    double result[25];
    const double q0 = q[0];
    const double q1 = q[1];
    const double q2 = q[2];
    const double q3 = q[3];
    const double q4 = q[4];
    const double q5 = q[5];
    const double p0 = profile[0];
    const double p1 = profile[1];
    const double p2 = profile[2];
    const double p3 = profile[3];
    const double p4 = profile[4];
    const double p5 = profile[5];
    const double p6 = profile[6];
    const double p7 = profile[7];
    const double p8 = profile[8];
    const double p9 = profile[9];
    const double p10 = profile[10];
    const double p11 = profile[11];
    const double v0 = cos(q2);
    const double v1 = pow(q0, 2);
    const double v2 = pow(q1, 2);
    const double v3 = v1 + v2;
    const double v4 = p1*v3;
    const double v5 = (1.0/16.0)*v0;
    const double v6 = p3*v3;
    const double v7 = (1.0/16.0)*v6;
    const double v8 = -1.0/2.0*p1*v0 + v0*v7 + v4*v5;
    const double v9 = -v8;
    const double v10 = sin(q2);
    const double v11 = p7*v3;
    const double v12 = (1.0/16.0)*v10;
    const double v13 = p9*v3;
    const double v14 = -1.0/2.0*p7*v10 + v11*v12 + v12*v13;
    const double v15 = -v14;
    const double v16 = q1*v15 + q1*v9;
    const double v17 = p1*v10;
    const double v18 = (1.0/8.0)*q1;
    const double v19 = p3*v10;
    const double v20 = p7*v0;
    const double v21 = p9*v0;
    const double v22 = (1.0/8.0)*q0;
    const double v23 = -q0*(-v17*v18 - v18*v19) - q0*(v18*v20 + v18*v21) + q1*(-v17*v22 - v19*v22) + q1*(v20*v22 + v21*v22);
    const double v24 = Q*v23;
    const double v25 = pow(v23, 2);
    const double v26 = v10*v4;
    const double v27 = (1.0/2.0)*p1*v10 - v10*v7 - 1.0/16.0*v26;
    const double v28 = (1.0/16.0)*v0*v11 + v13*v5 - 1.0/2.0*v20;
    const double v29 = Q*(q0*v27 + q0*v28) - q3;
    const double v30 = -v29;
    const double v31 = 1.0/q5;
    const double v32 = v30*v31;
    const double v33 = Q*v32;
    const double v34 = q0*v15 + q0*v9;
    const double v35 = Q*(q1*v27 + q1*v28) - q4;
    const double v36 = -v35;
    const double v37 = v16*v36;
    const double v38 = Q*v34;
    const double v39 = v31*v38;
    const double v40 = p0*v10;
    const double v41 = (1.0/2.0)*v40;
    const double v42 = p2*v10;
    const double v43 = p6*v0;
    const double v44 = (1.0/2.0)*v43;
    const double v45 = p8*v0;
    const double v46 = (1.0/2.0)*v45;
    const double v47 = (1.0/16.0)*q0;
    const double v48 = v3*v47;
    const double v49 = p10*v3;
    const double v50 = p2*v3;
    const double v51 = v10*v50;
    const double v52 = p4*v3;
    const double v53 = v10*v52;
    const double v54 = p8*v3;
    const double v55 = v5*v54;
    const double v56 = -v46 + v49*v5 + v55;
    const double v57 = v12*v50;
    const double v58 = (1.0/2.0)*p2*v10 - v12*v52 - v57;
    const double v59 = -q0*v41 - 1.0/2.0*q0*v42 + q0*v44 + q0*v46 + q0*v56 + q0*v58 - v0*v22*v54 - v0*v47*v49 + v22*v51 + v40*v48 - v43*v48 + v47*v53;
    const double v60 = pow(v59, 2);
    const double v61 = p0*v0;
    const double v62 = p6*v10;
    const double v63 = v3*v61;
    const double v64 = (1.0/4.0)*v0;
    const double v65 = v3*v62;
    const double v66 = (1.0/4.0)*v10;
    const double v67 = pow(v3, 2);
    const double v68 = (1.0/64.0)*v67;
    const double v69 = v10*v68;
    const double v70 = (1.0/32.0)*v67;
    const double v71 = v0*v70;
    const double v72 = v0*v68;
    const double v73 = v10*v70;
    const double v74 = p10*v69 + p2*v71 + p4*v72 + p8*v73 - v50*v64 - v54*v66 + v61*v68 + v61 + v62*v68 + v62 - 1.0/4.0*v63 - 1.0/4.0*v65;
    const double v75 = -v74;
    const double v76 = Q*v75;
    const double v77 = pow(q5, 2);
    const double v78 = sqrt(-pow(v30, 2) - pow(v36, 2) + v77 - 1);
    const double v79 = v75*v78;
    const double v80 = (1.0/16.0)*q1;
    const double v81 = (1.0/2.0)*p0*q1*v10 + (1.0/16.0)*p10*q1*v0*v3 + (1.0/2.0)*p2*q1*v10 + (1.0/16.0)*p6*q1*v0*v3 + (1.0/8.0)*p8*q1*v0*v3 - q1*v44 - q1*v46 - q1*v56 - q1*v58 - v18*v51 - v3*v40*v80 - v53*v80;
    const double v82 = v24*v31;
    const double v83 = v36*v81;
    const double v84 = Q*v31;
    const double v85 = v59*v84;
    const double v86 = 8*p7;
    const double v87 = -v0*(v11 + v13 - v86) + v10*(-8*p1 + v4 + v6);
    const double v88 = v31*v35;
    const double v89 = p1 + p3;
    const double v90 = v0*v89;
    const double v91 = p7 + p9;
    const double v92 = v10*v91;
    const double v93 = q0*v18;
    const double v94 = v93*(v90 + v92);
    const double v95 = (1.0/8.0)*v1;
    const double v96 = v14 + v8;
    const double v97 = v0*(-8*p2 + v50 + v52) + v10*(-8*p8 + v49 + v54);
    const double v98 = v0*v91 - v10*v89;
    const double v99 = p7*v0 + p9*v0 - v17 - v19;
    const double v100 = v29*v31;
    const double v101 = -v100*(v18*v98 - v18*v99) - v88*((1.0/8.0)*q0*v99 - v22*v98);
    const double v102 = 2*v10;
    const double v103 = 8*p0*v0 + 8*p2*v0 + 8*p6*v10 + 8*p8*v10 - 2*v0*v50 - v0*v52 - v10*v49 - v102*v54 - v63 - v65;
    const double v104 = v103*v47;
    const double v105 = v0*(p10 + p8);
    const double v106 = p2 + p4;
    const double v107 = v10*v106;
    const double v108 = p4*v10;
    const double v109 = v93*(v105 - v107) - v93*(p10*v0 + p6*v0 + 2*p8*v0 - v108 - v40 - 2*v42);
    const double v110 = (1.0/4.0)*v1;
    const double v111 = p10*v0;
    const double v112 = (1.0/16.0)*v3;
    const double v113 = v112*v40 - v112*v43 - v41 + v44 - v55 + v57;
    const double v114 = p5*v3;
    const double v115 = v0*(p11*v3 - 8*p9 + v13) - v10*(-8*p3 + v114 + v6);
    const double v116 = 8*p1*v10 + p11*v0*v3 + 8*p3*v10 + p7*v0*v3 + 2*p9*v0*v3 - v0*v86 - v10*v114 - v102*v6 - 8*v21 - v26;
    const double v117 = -v100*(v105*v95 - v107*v95 + v108*v95 + v110*v42 - v110*v45 - v111*v95 + v113 + v40*v95 - v43*v95) - v104 - v109*v88 + v31*v78*(v115*v47 - v116*v47) - v47*v97;
    const double v118 = v31*v78;
    const double v119 = v23*v36;
    const double v120 = v59*v78;
    const double v121 = v31*(v119 - v120) - v34;
    const double v122 = v23*v30;
    const double v123 = -v16 + v31*(-v122 + v78*v81);
    const double v124 = 1.0/v77;
    const double v125 = v30*v34;
    const double v126 = v125 + v37 + v79;
    const double v127 = v30*v59;
    const double v128 = v31*(v127 - v83) + v74;
    const double v129 = pow(v121, 2) + pow(v123, 2) - v124*pow(v126, 2) + pow(v128, 2);
    const double v130 = Q*q5;
    const double v131 = v129*v130;
    const double v132 = q5*((1.0/16.0)*q0*v31*v78*v97 + v101*v31*v36 - v117*v118 + v29*v31*(-v90*v95 - v92*v95 - v96) - v47*v87 - v88*v94) - v131*v30 - v16*v24 - v25*v33 + v33*pow(v34, 2) - v33*v60 + v37*v39 + v39*v79 + v59*v76 + v78*v81*v82 + v83*v85;
    const double v133 = 1.0/v78;
    const double v134 = rho*v130*v133;
    const double v135 = v31*v36;
    const double v136 = Q*v135;
    const double v137 = Q*v16;
    const double v138 = v137*v31;
    const double v139 = pow(v81, 2);
    const double v140 = v81*v84;
    const double v141 = (1.0/8.0)*v2;
    const double v142 = v103*v80;
    const double v143 = v100*v109 + v118*((1.0/16.0)*q1*v116 - v115*v80) + v142 + v80*v97 - v88*((1.0/8.0)*p10*v0*v2 + (1.0/8.0)*p6*v0*v2 + (1.0/4.0)*p8*v0*v2 + (1.0/8.0)*v10*v106*v2 - v105*v141 - v108*v141 - v113 - v141*v40 - 1.0/4.0*v2*v42);
    const double v144 = q5*((1.0/16.0)*q1*v31*v78*v97 - v100*v94 - v101*v30*v31 + v143*v31*v78 + v31*v35*(-v141*v90 - v141*v92 - v96) - v80*v87) + v120*v82 + v125*v138 + v127*v140 - v131*v36 - v136*v139 + v136*pow(v16, 2) - v136*v25 + v138*v79 + v24*v34 - v76*v81;
    const double v145 = v31*v76;
    const double v146 = Q*v118;
    const double v147 = (1.0/4.0)*v3;
    const double v148 = p1*v0;
    const double v149 = p7*v10;
    const double v150 = q5*(v100*v104 - v108*v68 + v111*v68 + v117*v32 - v118*((1.0/4.0)*p1*v0*v3 - p11*v69 + (1.0/4.0)*p3*v0*v3 - p3*v71 - p5*v72 + (1.0/4.0)*p7*v10*v3 + (1.0/4.0)*p9*v10*v3 - p9*v73 - v148*v68 - v148 - v149*v68 - v149) - v135*v143 + v142*v88 + v147*v40 - v147*v43 - v40*v68 - v40 - v42*v70 + v43*v68 + v43 + v45*v70 + v50*v66 - v54*v64) + v119*v85 + v122*v140 + v125*v145 - v131*v78 + v137*v81 - v139*v146 + v145*v37 - v146*v60 + v146*pow(v75, 2) - v38*v59;
    const double v151 = v132*v30 + v144*v36 + v150*v78;
    const double v152 = Q*rho;
    const double v153 = v151*v152;
    const double v154 = anomaly + v31;
    const double v155 = 1.0/(q5 + 1);
    const double v156 = v31*(anomaly + v155);
    const double v157 = anomaly*v155*(v23*v78 + v30*v81 + v36*v59);
    const double v158 = v152*v155;
    const double v159 = Q*v126;
    const double v160 = pow(Q, 2);
    const double v161 = rho*v77;
    const double v162 = v160*v161;
    const double v163 = v132*v34 + v144*v16 + v150*v75;
    const double v164 = pow(rho, 2);
    const double v165 = v160*v164;
    const double v166 = -v124*pow(v151, 2) + pow(v132, 2) + pow(v144, 2) + pow(v150, 2);
    const double v167 = 2*v121*v132 + 2*v123*v144 - 2*v124*v151*(v121*v30 + v123*v36 + v128*v78) + 2*v128*v150;
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
    result[3] = v132*v134;
    result[4] = v134*v144;
    result[5] = v133*v153;
    result[6] = -Q*(v154*v81 - v156*(v16*v78 - v36*v75) - v157*v32);
    result[7] = -Q*(-v135*v157 + v154*v59 - v156*(v30*v75 - v34*v78));
    result[8] = -Q*(-v118*v157 + v154*v23 - v156*(-v16*v30 + v34*v36));
    result[9] = -v158*(-v144*v78 + v150*v36);
    result[10] = v158*(-v132*v78 + v150*v30);
    result[11] = -v158*(-v132*v36 + v144*v30);
    result[12] = -v34;
    result[13] = -v16;
    result[14] = v74;
    result[15] = v81;
    result[16] = v59;
    result[17] = v23;
    result[18] = -v159*v31;
    result[19] = v153*v31;
    result[20] = v129*v162;
    result[21] = -rho*v159;
    result[22] = v163*v165;
    result[23] = v162*(rho*v167 + v129 + v164*v166);
    result[24] = v165*(v161*v166 + v163 + v167*v77);
    for (int i=0; i<25; ++i)
        if (!isfinite(result[i])) return CAVITY_NATIVE_NONFINITE;
    for (int i=0; i<25; ++i) output[i] = result[i];
    return CAVITY_NATIVE_OK;
}

// Explicit conservative BMT consumer: only omega3, no LL/Thomas/ledger work.
// Profile10 is Re0..4 then Im0..4, with the same uncharged source-mass units.
// This is a LAB-TIME rate; a spatial consumer applies gamma/uz once.
GPUFUN
int CavityNative_bmt(const double q[6], double Q, double anomaly,
        const double profile[10], double aperture, double omega[3]) {
    if (!isfinite(anomaly)) return CAVITY_NATIVE_ARGUMENT;
    double flow[6];
    int status = CavityNative_source(q, Q, profile, aperture, flow);
    if (status != CAVITY_NATIVE_OK) return status;
    double result[3];
    const double q0 = q[0];
    const double q1 = q[1];
    const double q2 = q[2];
    const double q3 = q[3];
    const double q4 = q[4];
    const double q5 = q[5];
    const double p0 = profile[0];
    const double p1 = profile[1];
    const double p2 = profile[2];
    const double p3 = profile[3];
    const double p4 = profile[4];
    const double p5 = profile[5];
    const double p6 = profile[6];
    const double p7 = profile[7];
    const double p8 = profile[8];
    const double p9 = profile[9];
    const double v0 = 1.0/q5;
    const double v1 = anomaly + v0;
    const double v2 = sin(q2);
    const double v3 = cos(q2);
    const double v4 = (1.0/2.0)*v3;
    const double v5 = p5*v4;
    const double v6 = p7*v4;
    const double v7 = pow(q0, 2) + pow(q1, 2);
    const double v8 = (1.0/16.0)*v7;
    const double v9 = v2*v8;
    const double v10 = p0*v9;
    const double v11 = (1.0/8.0)*q1;
    const double v12 = v11*v2;
    const double v13 = p2*v7;
    const double v14 = p4*v9;
    const double v15 = (1.0/2.0)*p2*v2 - p2*v9 - v14;
    const double v16 = v3*v8;
    const double v17 = p9*v16;
    const double v18 = p7*v16 + v17 - v6;
    const double v19 = (1.0/2.0)*p0*q1*v2 + (1.0/2.0)*p2*q1*v2 + (1.0/16.0)*p5*q1*v3*v7 + (1.0/8.0)*p7*q1*v3*v7 + (1.0/16.0)*p9*q1*v3*v7 - q1*v10 - q1*v14 - q1*v15 - q1*v18 - q1*v5 - q1*v6 - v12*v13;
    const double v20 = (1.0/2.0)*p1*v2 - p1*v9 - p3*v9;
    const double v21 = p6*v16 - p6*v4 + p8*v16;
    const double v22 = -Q*(q1*v20 + q1*v21) + q4;
    const double v23 = p0*v3;
    const double v24 = p5*v2;
    const double v25 = pow(v7, 2);
    const double v26 = (1.0/64.0)*v25;
    const double v27 = (1.0/32.0)*v25;
    const double v28 = (1.0/4.0)*p0*v3*v7 - p2*v27*v3 + (1.0/4.0)*p2*v3*v7 - p4*v26*v3 + (1.0/4.0)*p5*v2*v7 - p7*v2*v27 + (1.0/4.0)*p7*v2*v7 - p9*v2*v26 - v23*v26 - v23 - v24*v26 - v24;
    const double v29 = -p1*v16 + (1.0/2.0)*p1*v3 - p3*v16;
    const double v30 = (1.0/2.0)*p6*v2 - p6*v9 - p8*v9;
    const double v31 = q1*v29 + q1*v30;
    const double v32 = -Q*(q0*v20 + q0*v21) + q3;
    const double v33 = sqrt(pow(q5, 2) - pow(v22, 2) - pow(v32, 2) - 1);
    const double v34 = 1.0/(q5 + 1);
    const double v35 = v0*(anomaly + v34);
    const double v36 = v11*v3;
    const double v37 = (1.0/8.0)*q0;
    const double v38 = v2*v37;
    const double v39 = v3*v37;
    const double v40 = -q0*(-p1*v12 - p3*v12) - q0*(p6*v36 + p8*v36) + q1*(-p1*v38 - p3*v38) + q1*(p6*v39 + p8*v39);
    const double v41 = (1.0/2.0)*v2;
    const double v42 = -p0*q0*v41 - p2*q0*v41 - p5*q0*v16 - p7*v39*v7 + q0*v10 + q0*v14 + q0*v15 - q0*v17 + q0*v18 + q0*v5 + q0*v6 + v13*v38;
    const double v43 = anomaly*v0*v34*(v19*v32 + v22*v42 + v33*v40);
    const double v44 = q0*v29 + q0*v30;
    result[0] = -Q*(v1*v19 - v32*v43 - v35*(-v22*v28 + v31*v33));
    result[1] = -Q*(v1*v42 - v22*v43 - v35*(v28*v32 - v33*v44));
    result[2] = -Q*(v1*v40 - v33*v43 - v35*(v22*v44 - v31*v32));
    for (int i=0; i<3; ++i)
        if (!isfinite(result[i])) return CAVITY_NATIVE_NONFINITE;
    for (int i=0; i<3; ++i) omega[i] = result[i];
    return CAVITY_NATIVE_OK;
}

// Analytical S1 = partial_rho(reaction_source)|rho=0, already PER Z.
// No tiny-rho division, force amplification, anomaly or photon law.
// Inputs are the TRUE compact Re0..5/Im0..5 source jets and effective Q.
GPUFUN
int CavityNative_reaction_coefficient(const double q[6], double Q,
        const double profile[12], double aperture, double coefficient[6]) {
    for (int i=0; i<12; ++i)
        if (!isfinite(profile[i])) return CAVITY_NATIVE_NONFINITE;
    double orbit_profile[10], flow[6];
    for (int i=0; i<5; ++i) {
        orbit_profile[i] = profile[i];
        orbit_profile[i+5] = profile[i+6];
    }
    int status = CavityNative_source(q, Q, orbit_profile, aperture, flow);
    if (status != CAVITY_NATIVE_OK) return status;
    double result[6];
    const double q0 = q[0];
    const double q1 = q[1];
    const double q2 = q[2];
    const double q3 = q[3];
    const double q4 = q[4];
    const double q5 = q[5];
    const double p0 = profile[0];
    const double p1 = profile[1];
    const double p2 = profile[2];
    const double p3 = profile[3];
    const double p4 = profile[4];
    const double p5 = profile[5];
    const double p6 = profile[6];
    const double p7 = profile[7];
    const double p8 = profile[8];
    const double p9 = profile[9];
    const double p10 = profile[10];
    const double p11 = profile[11];
    const double v0 = cos(q2);
    const double v1 = pow(q0, 2);
    const double v2 = pow(q1, 2);
    const double v3 = v1 + v2;
    const double v4 = p1*v3;
    const double v5 = (1.0/16.0)*v0;
    const double v6 = p3*v3;
    const double v7 = (1.0/16.0)*v6;
    const double v8 = -1.0/2.0*p1*v0 + v0*v7 + v4*v5;
    const double v9 = -v8;
    const double v10 = sin(q2);
    const double v11 = p7*v3;
    const double v12 = (1.0/16.0)*v10;
    const double v13 = p9*v3;
    const double v14 = -1.0/2.0*p7*v10 + v11*v12 + v12*v13;
    const double v15 = -v14;
    const double v16 = q1*v15 + q1*v9;
    const double v17 = p1*v10;
    const double v18 = (1.0/8.0)*q1;
    const double v19 = p3*v10;
    const double v20 = p7*v0;
    const double v21 = p9*v0;
    const double v22 = (1.0/8.0)*q0;
    const double v23 = -q0*(-v17*v18 - v18*v19) - q0*(v18*v20 + v18*v21) + q1*(-v17*v22 - v19*v22) + q1*(v20*v22 + v21*v22);
    const double v24 = Q*v23;
    const double v25 = pow(v23, 2);
    const double v26 = v10*v4;
    const double v27 = (1.0/2.0)*p1*v10 - v10*v7 - 1.0/16.0*v26;
    const double v28 = (1.0/16.0)*v0*v11 + v13*v5 - 1.0/2.0*v20;
    const double v29 = Q*(q0*v27 + q0*v28) - q3;
    const double v30 = -v29;
    const double v31 = 1.0/q5;
    const double v32 = v30*v31;
    const double v33 = Q*v32;
    const double v34 = q0*v15 + q0*v9;
    const double v35 = Q*(q1*v27 + q1*v28) - q4;
    const double v36 = -v35;
    const double v37 = v16*v36;
    const double v38 = Q*v34;
    const double v39 = v31*v38;
    const double v40 = p0*v10;
    const double v41 = (1.0/2.0)*v40;
    const double v42 = p2*v10;
    const double v43 = p6*v0;
    const double v44 = (1.0/2.0)*v43;
    const double v45 = p8*v0;
    const double v46 = (1.0/2.0)*v45;
    const double v47 = (1.0/16.0)*q0;
    const double v48 = v3*v47;
    const double v49 = p10*v3;
    const double v50 = p2*v3;
    const double v51 = v10*v50;
    const double v52 = p4*v3;
    const double v53 = v10*v52;
    const double v54 = p8*v3;
    const double v55 = v5*v54;
    const double v56 = -v46 + v49*v5 + v55;
    const double v57 = v12*v50;
    const double v58 = (1.0/2.0)*p2*v10 - v12*v52 - v57;
    const double v59 = -q0*v41 - 1.0/2.0*q0*v42 + q0*v44 + q0*v46 + q0*v56 + q0*v58 - v0*v22*v54 - v0*v47*v49 + v22*v51 + v40*v48 - v43*v48 + v47*v53;
    const double v60 = pow(v59, 2);
    const double v61 = p0*v0;
    const double v62 = p6*v10;
    const double v63 = v3*v61;
    const double v64 = (1.0/4.0)*v0;
    const double v65 = v3*v62;
    const double v66 = (1.0/4.0)*v10;
    const double v67 = pow(v3, 2);
    const double v68 = (1.0/64.0)*v67;
    const double v69 = v10*v68;
    const double v70 = (1.0/32.0)*v67;
    const double v71 = v0*v70;
    const double v72 = v0*v68;
    const double v73 = v10*v70;
    const double v74 = p10*v69 + p2*v71 + p4*v72 + p8*v73 - v50*v64 - v54*v66 + v61*v68 + v61 + v62*v68 + v62 - 1.0/4.0*v63 - 1.0/4.0*v65;
    const double v75 = -v74;
    const double v76 = Q*v75;
    const double v77 = pow(q5, 2);
    const double v78 = sqrt(-pow(v30, 2) - pow(v36, 2) + v77 - 1);
    const double v79 = v75*v78;
    const double v80 = (1.0/16.0)*q1;
    const double v81 = (1.0/2.0)*p0*q1*v10 + (1.0/16.0)*p10*q1*v0*v3 + (1.0/2.0)*p2*q1*v10 + (1.0/16.0)*p6*q1*v0*v3 + (1.0/8.0)*p8*q1*v0*v3 - q1*v44 - q1*v46 - q1*v56 - q1*v58 - v18*v51 - v3*v40*v80 - v53*v80;
    const double v82 = v24*v31;
    const double v83 = v36*v81;
    const double v84 = Q*v31;
    const double v85 = v59*v84;
    const double v86 = 8*p7;
    const double v87 = -v0*(v11 + v13 - v86) + v10*(-8*p1 + v4 + v6);
    const double v88 = v31*v35;
    const double v89 = p1 + p3;
    const double v90 = v0*v89;
    const double v91 = p7 + p9;
    const double v92 = v10*v91;
    const double v93 = q0*v18;
    const double v94 = v93*(v90 + v92);
    const double v95 = (1.0/8.0)*v1;
    const double v96 = v14 + v8;
    const double v97 = v0*(-8*p2 + v50 + v52) + v10*(-8*p8 + v49 + v54);
    const double v98 = v0*v91 - v10*v89;
    const double v99 = p7*v0 + p9*v0 - v17 - v19;
    const double v100 = v29*v31;
    const double v101 = -v100*(v18*v98 - v18*v99) - v88*((1.0/8.0)*q0*v99 - v22*v98);
    const double v102 = 2*v10;
    const double v103 = 8*p0*v0 + 8*p2*v0 + 8*p6*v10 + 8*p8*v10 - 2*v0*v50 - v0*v52 - v10*v49 - v102*v54 - v63 - v65;
    const double v104 = v103*v47;
    const double v105 = v0*(p10 + p8);
    const double v106 = p2 + p4;
    const double v107 = v10*v106;
    const double v108 = p4*v10;
    const double v109 = v93*(v105 - v107) - v93*(p10*v0 + p6*v0 + 2*p8*v0 - v108 - v40 - 2*v42);
    const double v110 = (1.0/4.0)*v1;
    const double v111 = p10*v0;
    const double v112 = (1.0/16.0)*v3;
    const double v113 = v112*v40 - v112*v43 - v41 + v44 - v55 + v57;
    const double v114 = p5*v3;
    const double v115 = v0*(p11*v3 - 8*p9 + v13) - v10*(-8*p3 + v114 + v6);
    const double v116 = 8*p1*v10 + p11*v0*v3 + 8*p3*v10 + p7*v0*v3 + 2*p9*v0*v3 - v0*v86 - v10*v114 - v102*v6 - 8*v21 - v26;
    const double v117 = -v100*(v105*v95 - v107*v95 + v108*v95 + v110*v42 - v110*v45 - v111*v95 + v113 + v40*v95 - v43*v95) - v104 - v109*v88 + v31*v78*(v115*v47 - v116*v47) - v47*v97;
    const double v118 = v31*v78;
    const double v119 = v23*v36;
    const double v120 = v59*v78;
    const double v121 = v23*v30;
    const double v122 = v30*v34;
    const double v123 = v30*v59;
    const double v124 = Q*q5*(pow(-v16 + v31*(-v121 + v78*v81), 2) + pow(v31*(v119 - v120) - v34, 2) + pow(v31*(v123 - v83) + v74, 2) - pow(v122 + v37 + v79, 2)/v77);
    const double v125 = q5*((1.0/16.0)*q0*v31*v78*v97 + v101*v31*v36 - v117*v118 + v29*v31*(-v90*v95 - v92*v95 - v96) - v47*v87 - v88*v94) - v124*v30 - v16*v24 - v25*v33 + v33*pow(v34, 2) - v33*v60 + v37*v39 + v39*v79 + v59*v76 + v78*v81*v82 + v83*v85;
    const double v126 = Q/v78;
    const double v127 = q5*v126;
    const double v128 = v31*v36;
    const double v129 = Q*v128;
    const double v130 = Q*v16;
    const double v131 = v130*v31;
    const double v132 = pow(v81, 2);
    const double v133 = v81*v84;
    const double v134 = (1.0/8.0)*v2;
    const double v135 = v103*v80;
    const double v136 = v100*v109 + v118*((1.0/16.0)*q1*v116 - v115*v80) + v135 + v80*v97 - v88*((1.0/8.0)*p10*v0*v2 + (1.0/8.0)*p6*v0*v2 + (1.0/4.0)*p8*v0*v2 + (1.0/8.0)*v10*v106*v2 - v105*v134 - v108*v134 - v113 - v134*v40 - 1.0/4.0*v2*v42);
    const double v137 = q5*((1.0/16.0)*q1*v31*v78*v97 - v100*v94 - v101*v30*v31 + v136*v31*v78 + v31*v35*(-v134*v90 - v134*v92 - v96) - v80*v87) + v120*v82 + v122*v131 + v123*v133 - v124*v36 - v129*v132 + v129*pow(v16, 2) - v129*v25 + v131*v79 + v24*v34 - v76*v81;
    const double v138 = v31*v76;
    const double v139 = Q*v118;
    const double v140 = (1.0/4.0)*v3;
    const double v141 = p1*v0;
    const double v142 = p7*v10;
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
    result[3] = v125*v127;
    result[4] = v127*v137;
    result[5] = v126*(v125*v30 + v137*v36 + v78*(q5*(v100*v104 - v108*v68 + v111*v68 + v117*v32 - v118*((1.0/4.0)*p1*v0*v3 - p11*v69 + (1.0/4.0)*p3*v0*v3 - p3*v71 - p5*v72 + (1.0/4.0)*p7*v10*v3 + (1.0/4.0)*p9*v10*v3 - p9*v73 - v141*v68 - v141 - v142*v68 - v142) - v128*v136 + v135*v88 + v140*v40 - v140*v43 - v40*v68 - v40 - v42*v70 + v43*v68 + v43 + v45*v70 + v50*v66 - v54*v64) + v119*v85 + v121*v133 + v122*v138 - v124*v78 + v130*v81 - v132*v139 + v138*v37 - v139*v60 + v139*pow(v75, 2) - v38*v59));
    for (int i=0; i<6; ++i)
        if (!isfinite(result[i])) return CAVITY_NATIVE_NONFINITE;
    for (int i=0; i<6; ++i) coefficient[i] = result[i];
    return CAVITY_NATIVE_OK;
}
#endif
