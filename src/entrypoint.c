// Forward routine registration from C to Rust so the linker retains the
// extendr static library.
void R_init_dsemr_extendr(void *dll);
void register_extendr_panic_hook(void);

void R_init_DSEMr(void *dll) {
    register_extendr_panic_hook();
    R_init_dsemr_extendr(dll);
}

