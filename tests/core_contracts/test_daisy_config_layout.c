/* Compile-time contract of the Daisy memory budget.
 *
 * Including the config header IS the test: the pool-tiling _Static_asserts
 * in daisy_seed_config.h fire on any pool resize that no longer tiles the
 * 64 MB SDRAM exactly. The firmware target that normally compiles the header
 * needs the vendor BSP (absent on host/CI), so this host compile is what
 * keeps those asserts live in every test run.
 */
#include "daisy_seed_config.h"

#include <stdio.h>

int main(void) {
    /* Runtime restatement so the binary asserts something even if a future
     * refactor moves the _Static_asserts out of the header. */
    if (DAISY_WORK_POOL_OFFSET + DAISY_WORK_POOL_SIZE != DAISY_SDRAM_SIZE) {
        printf("FAIL: pools do not tile the SDRAM\n");
        return 1;
    }
    printf("OK: daisy pool layout tiles %lu MB SDRAM\n",
           (unsigned long)(DAISY_SDRAM_SIZE >> 20));
    return 0;
}
