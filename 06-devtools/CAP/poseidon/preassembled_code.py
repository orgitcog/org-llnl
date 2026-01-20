"""
A dictionary of assembled code used for these syscalls. The outer dict maps arch's to their code dicts, and the inner 
dicts map a string code name to its associated bytes

NOTE: this is only implemented for linux

NOTE: these were assembled by hand assuming:

  1. triton_executor.SCRATCH_SPACE_START_ADDRESS = 0xFFFF_FF00 (and thus, the 64-bit version is 0xFFFF_FFFF_FFFF_FF00)

Using the website: https://defuse.ca/online-x86-assembler.htm

Currently has the keys/codes:

    - 'mov_scratch_ret': moves the first long long value (4 bytes on 32-bit, 8 bytes 64-bit) at the starting scratch
      address into the return register (eax on X86, rax on X86_64)
    - 'mov_ret_ret': moves the stored at the memory location in return register into the return register (eax on X86, 
      rax on X86_64)
    - 'libc_enable_secure': computes the appropriate value for __libc_enable_secure (1 if True, 0 if False). Steps:
      
      1. Do getuid syscall, store into ebx
      2. Do geteuid syscall, store into ecx
      3. Subtract these two values
      4. Set edx to 0 if they are equal, 1 if not
      5. Do getgid syscall, store into ebx
      6. Do getegid syscall, store into ecx
      7. Subtract these two values
      8. Set ecx to 0 if they are equal, 1 if not

      x86 system call table can be found here: https://github.com/torvalds/linux/blob/v4.17/arch/x86/entry/syscalls/syscall_32.tbl#L17
      Fast/jumpless absolute value is computed with: abs(x) = (x ^ y) - y, where y = x >>> 31
"""
from triton import ARCH


ASSEMBLED_CODE_DICT = {
    ARCH.X86: {
        # mov eax, [0xffffff00]
        'mov_scratch_ret': b"\xA1\x00\xFF\xFF\xFF",

        # mov eax, [eax]
        'mov_ret_ret': b"\x8B\x00",

        # mov eax, 24      # getuid syscall
        # int 0x80
        # mov ebx, eax     # move getuid into ebx
        # mov eax, 49      # geteuid syscall
        # int 0x80
        # sub ebx, eax     # subtract getuid and geteuid. Should be 0 if same, something else if different
        # setne dl         # set edx to 0 if getuid and geteuid were the same, 1 if not
        # 
        # mov eax, 47      # getgid syscall
        # int 0x80
        # mov ebx, eax     # move getgid into ebx
        # mov eax, 50      # getegid syscall
        # int 0x80
        # sub ebx, eax     # subtract getgid and getegid. Should be 0 if same, something else if different
        # setne al         # set eax to 0 if getgid and getegid were the same, 1 if not
        # 
        # or eax, edx      # Values are now in eax/edx, set eax to 1 if either of them were 1 (or both)
        'libc_enable_secure': b"\xB8\x18\x00\x00\x00\xCD\x80\x89\xC3\xB8\x31\x00\x00\x00\xCD\x80\x29\xC3\x0F\x95\xC2\xB8\x2F\x00\x00\x00\xCD\x80\x89\xC3\xB8\x32\x00\x00\x00\xCD\x80\x29\xC3\x0F\x95\xC0\x09\xD0"
    },

    ARCH.X86_64: {
        # mov rax, [0xffffffffffffff00]
        'mov_scratch_ret': b"\x48\x8B\x04\x25\x00\xFF\xFF\xFF",

        # mov rax, [rax]
        'mov_ret_ret': b"\x48\x8B\x00",

        # mov eax, 102     # getuid syscall
        # syscall
        # mov ebx, eax     # move getuid into ebx
        # mov eax, 107     # geteuid syscall
        # syscall
        # sub ebx, eax     # subtract getuid and geteuid. Should be 0 if same, something else if different
        # setne dl         # set edx to 0 if getuid and geteuid were the same, 1 if not
        # 
        # mov eax, 104     # getgid syscall
        # syscall
        # mov ebx, eax     # move getgid into ebx
        # mov eax, 108     # getegid syscall
        # syscall
        # sub ebx, eax     # subtract getgid and getegid. Should be 0 if same, something else if different
        # setne al         # set eax to 0 if getgid and getegid were the same, 1 if not
        # 
        # or eax, edx      # Values are now in eax/edx, set eax to 1 if either of them were 1 (or both)
        'libc_enable_secure': b"\xB8\x66\x00\x00\x00\x0F\x05\x89\xC3\xB8\x6B\x00\x00\x00\x0F\x05\x29\xC3\x0F\x95\xC2\xB8\x68\x00\x00\x00\x0F\x05\x89\xC3\xB8\x6C\x00\x00\x00\x0F\x05\x29\xC3\x0F\x95\xC0\x09\xD0"
    },
}
