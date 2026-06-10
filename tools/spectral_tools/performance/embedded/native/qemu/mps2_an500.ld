/* QEMU mps2-an500 (Cortex-M7) layout for the counts rig (M7_PERF_MODEL_PLAN P2).
 * Map verified against `info mtree`: SSRAM1 4M @ 0x0 (code, vectors at 0),
 * SSRAM23 4M @ 0x20000000 (data/bss/stack), 16M @ 0x60000000 (mps.ram —
 * reserved for a future SDRAM-placement mirror of the Daisy segment buffer). */
ENTRY(Reset_Handler)

MEMORY
{
    CODE (rx)  : ORIGIN = 0x00000000, LENGTH = 4M
    DATA (rwx) : ORIGIN = 0x20000000, LENGTH = 4M
    BULK (rw)  : ORIGIN = 0x60000000, LENGTH = 16M
}

SECTIONS
{
    .isr_vector : { KEEP(*(.isr_vector)) } > CODE
    .text   : { *(.text*) }   > CODE
    .rodata : { *(.rodata*) } > CODE

    .data : ALIGN(4)
    {
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > DATA AT> CODE
    _sidata = LOADADDR(.data);

    .bss (NOLOAD) : ALIGN(4)
    {
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > DATA

    /* Optional SDRAM-mirror placement: objects put in .bulk_bss land in the
       0x60000000 region so the plugin's per-region byte buckets separate
       "SDRAM-like" traffic from working-set traffic. */
    .bulk_bss (NOLOAD) : ALIGN(4) { *(.bulk_bss*) } > BULK

    _estack = ORIGIN(DATA) + LENGTH(DATA);
}
