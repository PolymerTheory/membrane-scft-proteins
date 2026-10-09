// Parse D-HH:MM:SS or HH:MM:SS or MM:SS
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
// Query Slurm for time left on rank 0 only
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

// Call this at safe points. Sets sigg=7 if time left <= grace_seconds.
static inline void poll_time_and_set_sigg(MPI_Comm comm, int grace_seconds) {
    int left = -1, rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) left = slurm_seconds_left_local();
    MPI_Bcast(&left, 1, MPI_INT, 0, comm);
    if (left >= 0 && left <= grace_seconds) sigg = 7;
}

static inline void poll_time_and_set_sigg_local(int grace_seconds) {
        static time_t last_check = 0;
        time_t now = time(NULL);
        // only check every 30 s (or 60 s)
        if (now - last_check < 120) return;
        last_check = now;

        int left = slurm_seconds_left_local();
        if (left >= 0 && left <= grace_seconds) sigg = 7;
}

poll_time_and_set_sigg_local(3600); //put inside anderson main loop
