"""A manually checked CFG() for testing purposes

Built from the following C code, then run through ROSE for analysis:

```
include <stdio.h>

int search_char(char* s, char t) {
    char* base = s;
    while(*s && (*s != t)) {
        s++;
    }
    
    return (*s == 0 && t != 0) ? -1 : s-base;
}

int mutate(char* s, int seed) {
    char* base = s;
    int len = search_char(s,'\\0');
    if (len < 0) {
        return 1;
    }

    while(base && *s) {
        char* loc = base+((seed % *s) % (len));
        *loc = *s;
        s++;
    }
    return 0;
}
int main(int argc, char** argv) {
    char* s = argv[1];
    int colon_loc = search_char(s, ':');
    printf("colon_loc: %d\\n", colon_loc);
    printf("before: %s\\n", s);
    int mres = mutate(s, 31415);	
    printf("after : %s\\n", s);
    return 0;
}
```

"""
import os
from bincfg import CFGFunction, CFGBasicBlock, CFG, CFGEdge, EdgeType
from .fake_classes import FakeCFG, FakeCFGFunction


def get_manual_cfg(build_level):
    """Returns a manually built control flow graph
    
    Args:
        build_level (str): the level at which to build the cfg. Can be:

            - 'cfg': will build the full CFG() object
            - 'function': will build full CFGFunction's, but use a FakeCFG() object for the CFG
            - 'block': will build only CFGBasicBlock's. A FakeCFGFunction() object will be built as the functions that is
              simply an empty data structure to hold all the construction attributes being used, but with no processing

    Returns:
        dict[str, Any]: dictionary with the following keys/values:

            - 'blocks' (dict[int, CFGBasicBlock]): dictionary mapping basic block addresses to basic blocks
            - 'functions' (Union[dict[int, CFGFunction], dict[int, FakeCFGFunction]]): dictionary mapping function addresses
              to CFGFunctions (or, FakeCFGFunctions if we are not building functions)
            - 'cfg' (Union[CFG, object]): the CFG() object (if make_cfg=True) else a new object()
            - 'inputs' (list[CFGInputDataType]): input values that, when passed into CFG() constructor, should produce
              the exact same CFG() as that in 'cfg' (when build_level='cfg')
            - 'expected' (dict[str, Any]): dictionary of expected values. The following values are present:

                * 'sorted_block_order' (list[int]): list of basic block addresses in sorted order
                * 'sorted_func_order' (list[int]): list of function addresses in sorted order
                * 'num_blocks' (dict[int, int]): number of blocks per function, keys are function addresses
                * 'num_asm_lines_per_block' (dict[int, int]): number of asm lines per block, keys are block addresses
                * 'num_asm_lines_per_function' (dict[int, int]): number of asm lines per function, keys are function addresses
                * 'num_functions' (int): the number of functions
                * 'is_root_function' (dict[int, bool]): True if the function is a root function, keys are function addresses
                * 'is_extern_function' (dict[int, bool]): True if the function is an external function, keys are function addresses
                * 'is_intern_function' (dict[int, bool]): True if the function is an internal function, keys are function addresses
                * 'function_entry_block' (dict[int, int]): the address of the function entry block for each function, keys are function addresses
                * 'called_by' (dict[int, set[int]]): set of addresses of basic blocks that call each function, keys are function addresses
                * 'asm_counts_per_block' (dict[int, dict[str, int]]): dictionary of assembly line counts for each block, keys are block addresses
                * 'asm_counts_per_function' (dict[int, dict[str, int]]): dictionary of assembly line counts for each function, keys are function addresses
                * 'asm_counts' (dict[str, int]): dictinary of assembly line counts for entire CFG
    """

    if build_level not in ['cfg', 'function', 'block']:
        raise ValueError("Bad build_level: %s" % repr(build_level))
    
    func_type = CFGFunction if build_level in ['function', 'cfg'] else FakeCFGFunction

    # Create the cfg object. This cfg has 14 functions, 56 basic blocks, 65 edges, and 219 lines of assembly.
    __auto_cfg = CFG() if build_level in ['cfg'] else FakeCFG()

    # Building all functions. Dictionary maps integer address to CFGFunction() object
    __auto_functions = {
        4096: func_type(parent_cfg=__auto_cfg, address=4096, name='_init', is_extern_function=False, metadata={}),
        4160: func_type(parent_cfg=__auto_cfg, address=4160, name='__UNNAMED_FUNC_4160', is_extern_function=False, metadata={}),
        4176: func_type(parent_cfg=__auto_cfg, address=4176, name='printf@plt', is_extern_function=True, metadata={}),
        4192: func_type(parent_cfg=__auto_cfg, address=4192, name='_start', is_extern_function=False, metadata={}),
        4240: func_type(parent_cfg=__auto_cfg, address=4240, name='deregister_tm_clones', is_extern_function=False, metadata={}),
        4288: func_type(parent_cfg=__auto_cfg, address=4288, name='register_tm_clones', is_extern_function=False, metadata={}),
        4352: func_type(parent_cfg=__auto_cfg, address=4352, name='__do_global_dtors_aux', is_extern_function=False, metadata={}),
        4416: func_type(parent_cfg=__auto_cfg, address=4416, name='frame_dummy', is_extern_function=False, metadata={}),
        4425: func_type(parent_cfg=__auto_cfg, address=4425, name='search_char', is_extern_function=False, metadata={}),
        4514: func_type(parent_cfg=__auto_cfg, address=4514, name='mutate', is_extern_function=False, metadata={}),
        4655: func_type(parent_cfg=__auto_cfg, address=4655, name='main', is_extern_function=False, metadata={}),
        4816: func_type(parent_cfg=__auto_cfg, address=4816, name='__libc_csu_init', is_extern_function=False, metadata={}),
        4928: func_type(parent_cfg=__auto_cfg, address=4928, name='__libc_csu_fini', is_extern_function=False, metadata={}),
        4936: func_type(parent_cfg=__auto_cfg, address=4936, name='_fini', is_extern_function=False, metadata={}),
    }

    # Building basic blocks. Dictionary maps integer address to CFGBasicBlock() object
    __auto_blocks = {
        4096: CFGBasicBlock(parent_function=__auto_functions[4096], address=4096, asm_memory_addresses=[4096, 4100, 4104, 4111, 4114], metadata={}, asm_lines=[
            'nop',
            'sub    rsp, 0x08',
            'mov    rax, qword ds:[rip + 0x0000000000002fd9<12249,absolute=0x0000000000003fe8>]',
            'test   rax, rax',
            'je     0x0000000000001016<4118>',
        ]),
        4116: CFGBasicBlock(parent_function=__auto_functions[4096], address=4116, asm_memory_addresses=[4116], metadata={}, asm_lines=[
            'call   rax',
        ]),
        4118: CFGBasicBlock(parent_function=__auto_functions[4096], address=4118, asm_memory_addresses=[4118, 4122], metadata={}, asm_lines=[
            'add    rsp, 0x08',
            'ret',
        ]),
        4160: CFGBasicBlock(parent_function=__auto_functions[4160], address=4160, asm_memory_addresses=[4160, 4164], metadata={}, asm_lines=[
            'nop',
            'jmp    qword ds:[rip + 0x0000000000002fad<12205,absolute=0x0000000000003ff8>]',
        ]),
        4176: CFGBasicBlock(parent_function=__auto_functions[4176], address=4176, asm_memory_addresses=[4176, 4180], metadata={}, asm_lines=[
            'nop',
            'jmp    qword ds:[rip + 0x0000000000002f75<12149,absolute=0x0000000000003fd0>]',
        ]),
        4192: CFGBasicBlock(parent_function=__auto_functions[4192], address=4192, asm_memory_addresses=[4192, 4196, 4198, 4201, 4202, 4205, 4209, 4210, 4211, 4218, 4225, 4232], metadata={}, asm_lines=[
            'nop',
            'xor    ebp, ebp',
            'mov    r9, rdx',
            'pop    rsi',
            'mov    rdx, rsp',
            'and    rsp, 0xf0<-16>',
            'push   rax',
            'push   rsp',
            'lea    r8, [rip + 0x00000000000002c6<710,absolute=0x0000000000001340>]',
            'lea    rcx, [rip + 0x000000000000024f<591,absolute=0x00000000000012d0>]',
            'lea    rdi, [rip + 0x00000000000001a7<423,absolute=0x000000000000122f>]',
            'call   qword ds:[rip + 0x0000000000002f52<12114,absolute=0x0000000000003fe0>]',
        ]),
        4238: CFGBasicBlock(parent_function=__auto_functions[4192], address=4238, asm_memory_addresses=[4238], metadata={}, asm_lines=[
            'hlt',
        ]),
        4240: CFGBasicBlock(parent_function=__auto_functions[4240], address=4240, asm_memory_addresses=[4240, 4247, 4254, 4257], metadata={}, asm_lines=[
            'lea    rdi, [rip + 0x0000000000002f79<12153,absolute=0x0000000000004010>]',
            'lea    rax, [rip + 0x0000000000002f72<12146,absolute=0x0000000000004010>]',
            'cmp    rax, rdi',
            'je     0x00000000000010b8<4280>',
        ]),
        4259: CFGBasicBlock(parent_function=__auto_functions[4240], address=4259, asm_memory_addresses=[4259, 4266, 4269], metadata={}, asm_lines=[
            'mov    rax, qword ds:[rip + 0x0000000000002f2e<12078,absolute=0x0000000000003fd8>]',
            'test   rax, rax',
            'je     0x00000000000010b8<4280>',
        ]),
        4271: CFGBasicBlock(parent_function=__auto_functions[4240], address=4271, asm_memory_addresses=[4271], metadata={}, asm_lines=[
            'jmp    rax',
        ]),
        4273: CFGBasicBlock(parent_function=__auto_functions[4240], address=4273, asm_memory_addresses=[4273], metadata={}, asm_lines=[
            'nop    dword ds:[rax + 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>]',
        ]),
        4280: CFGBasicBlock(parent_function=__auto_functions[4240], address=4280, asm_memory_addresses=[4280], metadata={}, asm_lines=[
            'ret',
        ]),
        4288: CFGBasicBlock(parent_function=__auto_functions[4288], address=4288, asm_memory_addresses=[4288, 4295, 4302, 4305, 4308, 4312, 4316, 4319, 4322], metadata={}, asm_lines=[
            'lea    rdi, [rip + 0x0000000000002f49<12105,absolute=0x0000000000004010>]',
            'lea    rsi, [rip + 0x0000000000002f42<12098,absolute=0x0000000000004010>]',
            'sub    rsi, rdi',
            'mov    rax, rsi',
            'shr    rsi, 0x3f',
            'sar    rax, 0x03',
            'add    rsi, rax',
            'sar    rsi, 0x01',
            'je     0x00000000000010f8<4344>',
        ]),
        4324: CFGBasicBlock(parent_function=__auto_functions[4288], address=4324, asm_memory_addresses=[4324, 4331, 4334], metadata={}, asm_lines=[
            'mov    rax, qword ds:[rip + 0x0000000000002f05<12037,absolute=0x0000000000003ff0>]',
            'test   rax, rax',
            'je     0x00000000000010f8<4344>',
        ]),
        4336: CFGBasicBlock(parent_function=__auto_functions[4288], address=4336, asm_memory_addresses=[4336], metadata={}, asm_lines=[
            'jmp    rax',
        ]),
        4338: CFGBasicBlock(parent_function=__auto_functions[4288], address=4338, asm_memory_addresses=[4338], metadata={}, asm_lines=[
            'nop    word ds:[rax + rax + 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>]',
        ]),
        4344: CFGBasicBlock(parent_function=__auto_functions[4288], address=4344, asm_memory_addresses=[4344], metadata={}, asm_lines=[
            'ret',
        ]),
        4352: CFGBasicBlock(parent_function=__auto_functions[4352], address=4352, asm_memory_addresses=[4352, 4356, 4363], metadata={}, asm_lines=[
            'nop',
            'cmp    byte ds:[rip + 0x0000000000002f05<12037,absolute=0x0000000000004010>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>',
            'jne    0x0000000000001138<4408>',
        ]),
        4365: CFGBasicBlock(parent_function=__auto_functions[4352], address=4365, asm_memory_addresses=[4365, 4366, 4374, 4377], metadata={}, asm_lines=[
            'push   rbp',
            'cmp    qword ds:[rip + 0x0000000000002ee2<12002,absolute=0x0000000000003ff8>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>',
            'mov    rbp, rsp',
            'je     0x0000000000001127<4391>',
        ]),
        4379: CFGBasicBlock(parent_function=__auto_functions[4352], address=4379, asm_memory_addresses=[4379, 4386], metadata={}, asm_lines=[
            'mov    rdi, qword ds:[rip + 0x0000000000002ee6<12006,absolute=0x0000000000004008>]',
            'call   0x0000000000001040<4160>',
        ]),
        4391: CFGBasicBlock(parent_function=__auto_functions[4352], address=4391, asm_memory_addresses=[4391], metadata={}, asm_lines=[
            'call   0x0000000000001090<4240,(func)deregister_tm_clones>',
        ]),
        4396: CFGBasicBlock(parent_function=__auto_functions[4352], address=4396, asm_memory_addresses=[4396, 4403, 4404], metadata={}, asm_lines=[
            'mov    byte ds:[rip + 0x0000000000002edd<11997,absolute=0x0000000000004010>], 0x01',
            'pop    rbp',
            'ret',
        ]),
        4405: CFGBasicBlock(parent_function=__auto_functions[4352], address=4405, asm_memory_addresses=[4405], metadata={}, asm_lines=[
            'nop    dword ds:[rax]',
        ]),
        4408: CFGBasicBlock(parent_function=__auto_functions[4352], address=4408, asm_memory_addresses=[4408], metadata={}, asm_lines=[
            'ret',
        ]),
        4416: CFGBasicBlock(parent_function=__auto_functions[4416], address=4416, asm_memory_addresses=[4416, 4420], metadata={}, asm_lines=[
            'nop',
            'jmp    0x00000000000010c0<4288,(func)register_tm_clones>',
        ]),
        4425: CFGBasicBlock(parent_function=__auto_functions[4425], address=4425, asm_memory_addresses=[4425, 4429, 4430, 4433, 4437, 4439, 4442, 4446, 4450], metadata={}, asm_lines=[
            'nop',
            'push   rbp',
            'mov    rbp, rsp',
            'mov    qword ds:[rbp + 0xe8<-24>], rdi',
            'mov    eax, esi',
            'mov    byte ds:[rbp + 0xe4<-28>], al',
            'mov    rax, qword ds:[rbp + 0xe8<-24>]',
            'mov    qword ds:[rbp + 0xf8<-8>], rax',
            'jmp    0x0000000000001169<4457>',
        ]),
        4452: CFGBasicBlock(parent_function=__auto_functions[4425], address=4452, asm_memory_addresses=[4452], metadata={}, asm_lines=[
            'add    qword ds:[rbp + 0xe8<-24>], 0x01',
        ]),
        4457: CFGBasicBlock(parent_function=__auto_functions[4425], address=4457, asm_memory_addresses=[4457, 4461, 4464, 4466], metadata={}, asm_lines=[
            'mov    rax, qword ds:[rbp + 0xe8<-24>]',
            'movzx  eax, byte ds:[rax]',
            'test   al, al',
            'je     0x0000000000001180<4480>',
        ]),
        4468: CFGBasicBlock(parent_function=__auto_functions[4425], address=4468, asm_memory_addresses=[4468, 4472, 4475, 4478], metadata={}, asm_lines=[
            'mov    rax, qword ds:[rbp + 0xe8<-24>]',
            'movzx  eax, byte ds:[rax]',
            'cmp    byte ds:[rbp + 0xe4<-28>], al',
            'jne    0x0000000000001164<4452>',
        ]),
        4480: CFGBasicBlock(parent_function=__auto_functions[4425], address=4480, asm_memory_addresses=[4480, 4484, 4487, 4489], metadata={}, asm_lines=[
            'mov    rax, qword ds:[rbp + 0xe8<-24>]',
            'movzx  eax, byte ds:[rax]',
            'test   al, al',
            'jne    0x0000000000001191<4497>',
        ]),
        4491: CFGBasicBlock(parent_function=__auto_functions[4425], address=4491, asm_memory_addresses=[4491, 4495], metadata={}, asm_lines=[
            'cmp    byte ds:[rbp + 0xe4<-28>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>',
            'jne    0x000000000000119b<4507>',
        ]),
        4497: CFGBasicBlock(parent_function=__auto_functions[4425], address=4497, asm_memory_addresses=[4497, 4501, 4505], metadata={}, asm_lines=[
            'mov    rax, qword ds:[rbp + 0xe8<-24>]',
            'sub    rax, qword ds:[rbp + 0xf8<-8>]',
            'jmp    0x00000000000011a0<4512>',
        ]),
        4507: CFGBasicBlock(parent_function=__auto_functions[4425], address=4507, asm_memory_addresses=[4507], metadata={}, asm_lines=[
            'mov    eax, 0xffffffff<-1>',
        ]),
        4512: CFGBasicBlock(parent_function=__auto_functions[4425], address=4512, asm_memory_addresses=[4512, 4513], metadata={}, asm_lines=[
            'pop    rbp',
            'ret',
        ]),
        4514: CFGBasicBlock(parent_function=__auto_functions[4514], address=4514, asm_memory_addresses=[4514, 4518, 4519, 4522, 4526, 4530, 4533, 4537, 4541, 4545, 4550, 4553], metadata={}, asm_lines=[
            'nop',
            'push   rbp',
            'mov    rbp, rsp',
            'sub    rsp, 0x30',
            'mov    qword ds:[rbp + 0xd8<-40>], rdi',
            'mov    dword ds:[rbp + 0xd4<-44>], esi',
            'mov    rax, qword ds:[rbp + 0xd8<-40>]',
            'mov    qword ds:[rbp + 0xf0<-16>], rax',
            'mov    rax, qword ds:[rbp + 0xd8<-40>]',
            'mov    esi, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>',
            'mov    rdi, rax',
            'call   0x0000000000001149<4425,(func)search_char>',
        ]),
        4558: CFGBasicBlock(parent_function=__auto_functions[4514], address=4558, asm_memory_addresses=[4558, 4561, 4565], metadata={}, asm_lines=[
            'mov    dword ds:[rbp + 0xec<-20>], eax',
            'cmp    dword ds:[rbp + 0xec<-20>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>',
            'jns    0x0000000000001216<4630>',
        ]),
        4567: CFGBasicBlock(parent_function=__auto_functions[4514], address=4567, asm_memory_addresses=[4567, 4572], metadata={}, asm_lines=[
            'mov    eax, 0x00000001',
            'jmp    0x000000000000122d<4653>',
        ]),
        4574: CFGBasicBlock(parent_function=__auto_functions[4514], address=4574, asm_memory_addresses=[4574, 4578, 4581, 4584, 4587, 4588, 4590, 4592, 4593, 4596, 4598, 4601, 4605, 4608, 4612, 4616, 4619, 4623, 4625], metadata={}, asm_lines=[
            'mov    rax, qword ds:[rbp + 0xd8<-40>]',
            'movzx  eax, byte ds:[rax]',
            'movsx  ecx, al',
            'mov    eax, dword ds:[rbp + 0xd4<-44>]',
            'cdq',
            'idiv   ecx',
            'mov    eax, edx',
            'cdq',
            'idiv   dword ds:[rbp + 0xec<-20>]',
            'mov    eax, edx',
            'movsxd rdx, eax',
            'mov    rax, qword ds:[rbp + 0xf0<-16>]',
            'add    rax, rdx',
            'mov    qword ds:[rbp + 0xf8<-8>], rax',
            'mov    rax, qword ds:[rbp + 0xd8<-40>]',
            'movzx  edx, byte ds:[rax]',
            'mov    rax, qword ds:[rbp + 0xf8<-8>]',
            'mov    byte ds:[rax], dl',
            'add    qword ds:[rbp + 0xd8<-40>], 0x01',
        ]),
        4630: CFGBasicBlock(parent_function=__auto_functions[4514], address=4630, asm_memory_addresses=[4630, 4635], metadata={}, asm_lines=[
            'cmp    qword ds:[rbp + 0xf0<-16>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>',
            'je     0x0000000000001228<4648>',
        ]),
        4637: CFGBasicBlock(parent_function=__auto_functions[4514], address=4637, asm_memory_addresses=[4637, 4641, 4644, 4646], metadata={}, asm_lines=[
            'mov    rax, qword ds:[rbp + 0xd8<-40>]',
            'movzx  eax, byte ds:[rax]',
            'test   al, al',
            'jne    0x00000000000011de<4574>',
        ]),
        4648: CFGBasicBlock(parent_function=__auto_functions[4514], address=4648, asm_memory_addresses=[4648], metadata={}, asm_lines=[
            'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>',
        ]),
        4653: CFGBasicBlock(parent_function=__auto_functions[4514], address=4653, asm_memory_addresses=[4653, 4654], metadata={}, asm_lines=[
            'leave',
            'ret',
        ]),
        4655: CFGBasicBlock(parent_function=__auto_functions[4655], address=4655, asm_memory_addresses=[4655, 4659, 4660, 4663, 4667, 4670, 4674, 4678, 4682, 4686, 4690, 4695, 4698], metadata={}, asm_lines=[
            'nop',
            'push   rbp',
            'mov    rbp, rsp',
            'sub    rsp, 0x20',
            'mov    dword ds:[rbp + 0xec<-20>], edi',
            'mov    qword ds:[rbp + 0xe0<-32>], rsi',
            'mov    rax, qword ds:[rbp + 0xe0<-32>]',
            'mov    rax, qword ds:[rax + 0x08]',
            'mov    qword ds:[rbp + 0xf8<-8>], rax',
            'mov    rax, qword ds:[rbp + 0xf8<-8>]',
            'mov    esi, 0x0000003a',
            'mov    rdi, rax',
            'call   0x0000000000001149<4425,(func)search_char>',
        ]),
        4703: CFGBasicBlock(parent_function=__auto_functions[4655], address=4703, asm_memory_addresses=[4703, 4706, 4709, 4711, 4718, 4723], metadata={}, asm_lines=[
            'mov    dword ds:[rbp + 0xf0<-16>], eax',
            'mov    eax, dword ds:[rbp + 0xf0<-16>]',
            'mov    esi, eax',
            'lea    rdi, [rip + 0x0000000000000d96<3478,absolute=0x0000000000002004>]',
            'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>',
            'call   0x0000000000001050<4176>',
        ]),
        4728: CFGBasicBlock(parent_function=__auto_functions[4655], address=4728, asm_memory_addresses=[4728, 4732, 4735, 4742, 4747], metadata={}, asm_lines=[
            'mov    rax, qword ds:[rbp + 0xf8<-8>]',
            'mov    rsi, rax',
            'lea    rdi, [rip + 0x0000000000000d8d<3469,absolute=0x0000000000002013>]',
            'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>',
            'call   0x0000000000001050<4176>',
        ]),
        4752: CFGBasicBlock(parent_function=__auto_functions[4655], address=4752, asm_memory_addresses=[4752, 4756, 4761, 4764], metadata={}, asm_lines=[
            'mov    rax, qword ds:[rbp + 0xf8<-8>]',
            'mov    esi, 0x00007ab7<31415>',
            'mov    rdi, rax',
            'call   0x00000000000011a2<4514,(func)mutate>',
        ]),
        4769: CFGBasicBlock(parent_function=__auto_functions[4655], address=4769, asm_memory_addresses=[4769, 4772, 4776, 4779, 4786, 4791], metadata={}, asm_lines=[
            'mov    dword ds:[rbp + 0xf4<-12>], eax',
            'mov    rax, qword ds:[rbp + 0xf8<-8>]',
            'mov    rsi, rax',
            'lea    rdi, [rip + 0x0000000000000d6d<3437,absolute=0x000000000000201f>]',
            'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>',
            'call   0x0000000000001050<4176>',
        ]),
        4796: CFGBasicBlock(parent_function=__auto_functions[4655], address=4796, asm_memory_addresses=[4796, 4801, 4802], metadata={}, asm_lines=[
            'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>',
            'leave',
            'ret',
        ]),
        4816: CFGBasicBlock(parent_function=__auto_functions[4816], address=4816, asm_memory_addresses=[4816, 4820, 4822, 4829, 4831, 4834, 4836, 4839, 4841, 4844, 4845, 4852, 4853, 4856, 4860], metadata={}, asm_lines=[
            'nop',
            'push   r15',
            'lea    r15, [rip + 0x0000000000002adb<10971,absolute=0x0000000000003db8>]',
            'push   r14',
            'mov    r14, rdx',
            'push   r13',
            'mov    r13, rsi',
            'push   r12',
            'mov    r12d, edi',
            'push   rbp',
            'lea    rbp, [rip + 0x0000000000002acc<10956,absolute=0x0000000000003dc0>]',
            'push   rbx',
            'sub    rbp, r15',
            'sub    rsp, 0x08',
            'call   0x0000000000001000<4096,(func)_init>',
        ]),
        4865: CFGBasicBlock(parent_function=__auto_functions[4816], address=4865, asm_memory_addresses=[4865, 4869], metadata={}, asm_lines=[
            'sar    rbp, 0x03',
            'je     0x0000000000001326<4902>',
        ]),
        4871: CFGBasicBlock(parent_function=__auto_functions[4816], address=4871, asm_memory_addresses=[4871, 4873], metadata={}, asm_lines=[
            'xor    ebx, ebx',
            'nop    dword ds:[rax + 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>]',
        ]),
        4880: CFGBasicBlock(parent_function=__auto_functions[4816], address=4880, asm_memory_addresses=[4880, 4883, 4886, 4889], metadata={}, asm_lines=[
            'mov    rdx, r14',
            'mov    rsi, r13',
            'mov    edi, r12d',
            'call   qword ds:[r15 + rbx*0x08]',
        ]),
        4893: CFGBasicBlock(parent_function=__auto_functions[4816], address=4893, asm_memory_addresses=[4893, 4897, 4900], metadata={}, asm_lines=[
            'add    rbx, 0x01',
            'cmp    rbp, rbx',
            'jne    0x0000000000001310<4880>',
        ]),
        4902: CFGBasicBlock(parent_function=__auto_functions[4816], address=4902, asm_memory_addresses=[4902, 4906, 4907, 4908, 4910, 4912, 4914, 4916], metadata={}, asm_lines=[
            'add    rsp, 0x08',
            'pop    rbx',
            'pop    rbp',
            'pop    r12',
            'pop    r13',
            'pop    r14',
            'pop    r15',
            'ret',
        ]),
        4928: CFGBasicBlock(parent_function=__auto_functions[4928], address=4928, asm_memory_addresses=[4928, 4932], metadata={}, asm_lines=[
            'nop',
            'ret',
        ]),
        4936: CFGBasicBlock(parent_function=__auto_functions[4936], address=4936, asm_memory_addresses=[4936, 4940, 4944, 4948], metadata={}, asm_lines=[
            'nop',
            'sub    rsp, 0x08',
            'add    rsp, 0x08',
            'ret',
        ]),
    }

    # Building all edges
    __auto_blocks[4096].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4096], to_block=__auto_blocks[4118], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4096], to_block=__auto_blocks[4116], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4116].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4116], to_block=__auto_blocks[4118], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4118].edges_out = set([
        
    ])

    __auto_blocks[4160].edges_out = set([
        
    ])

    __auto_blocks[4176].edges_out = set([
        
    ])

    __auto_blocks[4192].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4192], to_block=__auto_blocks[4238], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4238].edges_out = set([
        
    ])

    __auto_blocks[4240].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4240], to_block=__auto_blocks[4259], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4240], to_block=__auto_blocks[4280], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4259].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4259], to_block=__auto_blocks[4271], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4259], to_block=__auto_blocks[4280], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4271].edges_out = set([
        
    ])

    __auto_blocks[4273].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4273], to_block=__auto_blocks[4280], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4280].edges_out = set([
        
    ])

    __auto_blocks[4288].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4288], to_block=__auto_blocks[4324], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4288], to_block=__auto_blocks[4344], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4324].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4324], to_block=__auto_blocks[4344], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4324], to_block=__auto_blocks[4336], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4336].edges_out = set([
        
    ])

    __auto_blocks[4338].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4338], to_block=__auto_blocks[4344], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4344].edges_out = set([
        
    ])

    __auto_blocks[4352].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4352], to_block=__auto_blocks[4408], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4352], to_block=__auto_blocks[4365], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4365].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4365], to_block=__auto_blocks[4391], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4365], to_block=__auto_blocks[4379], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4379].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4379], to_block=__auto_blocks[4160], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4379], to_block=__auto_blocks[4391], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4391].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4391], to_block=__auto_blocks[4240], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4391], to_block=__auto_blocks[4396], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4396].edges_out = set([
        
    ])

    __auto_blocks[4405].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4405], to_block=__auto_blocks[4408], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4408].edges_out = set([
        
    ])

    __auto_blocks[4416].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4416], to_block=__auto_blocks[4288], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4425].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4425], to_block=__auto_blocks[4457], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4452].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4452], to_block=__auto_blocks[4457], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4457].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4457], to_block=__auto_blocks[4468], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4457], to_block=__auto_blocks[4480], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4468].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4468], to_block=__auto_blocks[4480], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4468], to_block=__auto_blocks[4452], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4480].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4480], to_block=__auto_blocks[4491], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4480], to_block=__auto_blocks[4497], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4491].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4491], to_block=__auto_blocks[4507], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4491], to_block=__auto_blocks[4497], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4497].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4497], to_block=__auto_blocks[4512], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4507].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4507], to_block=__auto_blocks[4512], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4512].edges_out = set([
        
    ])

    __auto_blocks[4514].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4514], to_block=__auto_blocks[4425], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4514], to_block=__auto_blocks[4558], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4558].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4558], to_block=__auto_blocks[4630], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4558], to_block=__auto_blocks[4567], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4567].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4567], to_block=__auto_blocks[4653], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4574].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4574], to_block=__auto_blocks[4630], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4630].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4630], to_block=__auto_blocks[4637], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4630], to_block=__auto_blocks[4648], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4637].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4637], to_block=__auto_blocks[4574], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4637], to_block=__auto_blocks[4648], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4648].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4648], to_block=__auto_blocks[4653], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4653].edges_out = set([
        
    ])

    __auto_blocks[4655].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4655], to_block=__auto_blocks[4425], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4655], to_block=__auto_blocks[4703], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4703].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4703], to_block=__auto_blocks[4176], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4703], to_block=__auto_blocks[4728], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4728].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4728], to_block=__auto_blocks[4176], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4728], to_block=__auto_blocks[4752], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4752].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4752], to_block=__auto_blocks[4514], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4752], to_block=__auto_blocks[4769], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4769].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4769], to_block=__auto_blocks[4796], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4769], to_block=__auto_blocks[4176], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[4796].edges_out = set([
        
    ])

    __auto_blocks[4816].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4816], to_block=__auto_blocks[4865], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4816], to_block=__auto_blocks[4096], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[4865].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4865], to_block=__auto_blocks[4902], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4865], to_block=__auto_blocks[4871], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4871].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4871], to_block=__auto_blocks[4880], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4880].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4880], to_block=__auto_blocks[4893], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4893].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4893], to_block=__auto_blocks[4902], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4893], to_block=__auto_blocks[4880], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4902].edges_out = set([
        
    ])

    __auto_blocks[4928].edges_out = set([
        
    ])

    __auto_blocks[4936].edges_out = set([
        
    ])

    # Set the edges_in on the blocks
    for b in __auto_blocks.values():
        for e in b.edges_out:
            e.to_block.edges_in.add(CFGEdge(b, e.to_block, e.edge_type))

    # Adding basic blocks to their associated functions
    __auto_functions[4096].blocks = [
        __auto_blocks[4096],
        __auto_blocks[4116],
        __auto_blocks[4118],
    ]

    __auto_functions[4160].blocks = [
        __auto_blocks[4160],
    ]

    __auto_functions[4176].blocks = [
        __auto_blocks[4176],
    ]

    __auto_functions[4192].blocks = [
        __auto_blocks[4192],
        __auto_blocks[4238],
    ]

    __auto_functions[4240].blocks = [
        __auto_blocks[4240],
        __auto_blocks[4259],
        __auto_blocks[4271],
        __auto_blocks[4273],
        __auto_blocks[4280],
    ]

    __auto_functions[4288].blocks = [
        __auto_blocks[4288],
        __auto_blocks[4324],
        __auto_blocks[4336],
        __auto_blocks[4338],
        __auto_blocks[4344],
    ]

    __auto_functions[4352].blocks = [
        __auto_blocks[4352],
        __auto_blocks[4365],
        __auto_blocks[4379],
        __auto_blocks[4391],
        __auto_blocks[4396],
        __auto_blocks[4405],
        __auto_blocks[4408],
    ]

    __auto_functions[4416].blocks = [
        __auto_blocks[4416],
    ]

    __auto_functions[4425].blocks = [
        __auto_blocks[4425],
        __auto_blocks[4452],
        __auto_blocks[4457],
        __auto_blocks[4468],
        __auto_blocks[4480],
        __auto_blocks[4491],
        __auto_blocks[4497],
        __auto_blocks[4507],
        __auto_blocks[4512],
    ]

    __auto_functions[4514].blocks = [
        __auto_blocks[4514],
        __auto_blocks[4558],
        __auto_blocks[4567],
        __auto_blocks[4574],
        __auto_blocks[4630],
        __auto_blocks[4637],
        __auto_blocks[4648],
        __auto_blocks[4653],
    ]

    __auto_functions[4655].blocks = [
        __auto_blocks[4655],
        __auto_blocks[4703],
        __auto_blocks[4728],
        __auto_blocks[4752],
        __auto_blocks[4769],
        __auto_blocks[4796],
    ]

    __auto_functions[4816].blocks = [
        __auto_blocks[4816],
        __auto_blocks[4865],
        __auto_blocks[4871],
        __auto_blocks[4880],
        __auto_blocks[4893],
        __auto_blocks[4902],
    ]

    __auto_functions[4928].blocks = [
        __auto_blocks[4928],
    ]

    __auto_functions[4936].blocks = [
        __auto_blocks[4936],
    ]

    # The expected values for this cfg
    expected = {
      'sorted_func_order': [4096, 4160, 4176, 4192, 4240, 4288, 4352, 4416, 4425, 4514, 4655, 4816, 4928, 4936],
      'sorted_block_order': [4096, 4116, 4118, 4160, 4176, 4192, 4238, 4240, 4259, 4271, 4273, 4280, 4288, 4324, 4336, 4338, 4344, 4352, 4365, 4379, 4391, 4396, 4405, 4408, 4416, 4425, 4452, 4457, 4468, 4480, 4491, 4497, 4507, 4512, 4514, 4558, 4567, 4574, 4630, 4637, 4648, 4653, 4655, 4703, 4728, 4752, 4769, 4796, 4816, 4865, 4871, 4880, 4893, 4902, 4928, 4936],
      'architecture': 'x86',
      'num_blocks': {4096: 3, 4160: 1, 4176: 1, 4192: 2, 4240: 5, 4288: 5, 4352: 7, 4416: 1, 4425: 9, 4514: 8, 4655: 6, 4816: 6, 4928: 1, 4936: 1},
      'num_asm_lines_per_block': {4096: 5, 4116: 1, 4118: 2, 4160: 2, 4176: 2, 4192: 12, 4238: 1, 4240: 4, 4259: 3, 4271: 1, 4273: 1, 4280: 1, 4288: 9, 4324: 3, 4336: 1, 4338: 1, 4344: 1, 4352: 3, 4365: 4, 4379: 2, 4391: 1, 4396: 3, 4405: 1, 4408: 1, 4416: 2, 4425: 9, 4452: 1, 4457: 4, 4468: 4, 4480: 4, 4491: 2, 4497: 3, 4507: 1, 4512: 2, 4514: 12, 4558: 3, 4567: 2, 4574: 19, 4630: 2, 4637: 4, 4648: 1, 4653: 2, 4655: 13, 4703: 6, 4728: 5, 4752: 4, 4769: 6, 4796: 3, 4816: 15, 4865: 2, 4871: 2, 4880: 4, 4893: 3, 4902: 8, 4928: 2, 4936: 4},
      'num_asm_lines_per_function': {4096: 8, 4160: 2, 4176: 2, 4192: 13, 4240: 10, 4288: 15, 4352: 15, 4416: 2, 4425: 30, 4514: 45, 4655: 37, 4816: 34, 4928: 2, 4936: 4},
      'num_functions': 14,
      'is_root_function': {4096: False, 4160: False, 4176: False, 4192: True, 4240: False, 4288: True, 4352: True, 4416: True, 4425: False, 4514: False, 4655: True, 4816: True, 4928: True, 4936: True},
      'is_recursive': {4096: False, 4160: False, 4176: False, 4192: False, 4240: False, 4288: False, 4352: False, 4416: False, 4425: False, 4514: False, 4655: False, 4816: False, 4928: False, 4936: False},
      'is_extern_function': {4096: False, 4160: False, 4176: True, 4192: False, 4240: False, 4288: False, 4352: False, 4416: False, 4425: False, 4514: False, 4655: False, 4816: False, 4928: False, 4936: False},
      'is_intern_function': {4096: True, 4160: True, 4176: False, 4192: True, 4240: True, 4288: True, 4352: True, 4416: True, 4425: True, 4514: True, 4655: True, 4816: True, 4928: True, 4936: True},
      'function_entry_block': {4096: 4096, 4160: 4160, 4176: 4176, 4192: 4192, 4240: 4240, 4288: 4288, 4352: 4352, 4416: 4416, 4425: 4425, 4514: 4514, 4655: 4655, 4816: 4816, 4928: 4928, 4936: 4936},
      'called_by': {4096: {4816}, 4160: {4379}, 4176: {4728, 4769, 4703}, 4192: set(), 4240: {4391}, 4288: set(), 4352: set(), 4416: set(), 4425: {4514, 4655}, 4514: {4752}, 4655: set(), 4816: set(), 4928: set(), 4936: set()},
      'function_hashes': {4096: 439172561004494102, 4160: 2226954711652702726, 4176: 38249409829118922, 4192: 2042503580022717622, 4240: 503703899796048437, 4288: 1744112140822227037, 4352: 2292357954646068411, 4416: 555562848796908185, 4425: 225605470578832735, 4514: 1119246084232895455, 4655: 1025589057692202207, 4816: 1177351854230011188, 4928: 1204923052535462881, 4936: 1043929603289091238},
      'block_hashes': {4096: 1565488076614152736, 4116: 2082162106127116742, 4118: 1498010728489710195, 4160: 1034915747952645316, 4176: 889000240665788365, 4192: 47747070843908631, 4238: 1733495061542071184, 4240: 1058067733596575428, 4259: 554213819008755358, 4271: 997552277369840808, 4273: 1331304933764275022, 4280: 74226675414060432, 4288: 626704969159444712, 4324: 1925367347447909427, 4336: 103660973604748862, 4338: 1358657175472023732, 4344: 1627784235096510757, 4352: 1500528524810510781, 4365: 141495747550943620, 4379: 1145319147612364580, 4391: 835150270072828084, 4396: 760264885376148879, 4405: 1120851607511439082, 4408: 683741836203387455, 4416: 75983617151313020, 4425: 1294022233363635853, 4452: 1509612827754248094, 4457: 1949663822187445281, 4468: 275111377082732221, 4480: 742645782903866710, 4491: 695905174477998400, 4497: 1668284436379290458, 4507: 497981739934380807, 4512: 633472653301466370, 4514: 1783423702362301065, 4558: 2122133376345504991, 4567: 2068403400721506200, 4574: 1553012402294834048, 4630: 1199367959671453789, 4637: 754918601605012432, 4648: 776343415401491, 4653: 974674477942020638, 4655: 1762653010252578577, 4703: 443853734367165119, 4728: 1099539015939882841, 4752: 259814106682938966, 4769: 1193071880753883777, 4796: 274649367180802496, 4816: 1599065879961768706, 4865: 2009398158284905529, 4871: 253294751146395741, 4880: 790527230682649427, 4893: 1117861546914567758, 4902: 1820528438940898469, 4928: 2175583907841071789, 4936: 657480897710522952},
      'cfg_hash': 345096326539219619,
      'memcfg_hashes': {'base_norm-op': 687432822938721222, 'base_norm-inst': 1258230514261818778, 'innereye-op': 421872215730983417, 'innereye-inst': 820800979529640689, 'safe-op': 636635059827283480, 'safe-inst': 1876853684191917046, 'deepbindiff-op': 1556133027177905737, 'deepbindiff-inst': 570136207548907837, 'deepsemantic-op': 151003745820316094, 'deepsemantic-inst': 959325029005630261, 'compressed_stats-op': 1475474773087009784, 'compressed_stats-inst': 1219583163932092313, 'hpcdata-op': 290032171087232721, 'hpcdata-inst': 700212233917802347},
      'asm_counts_per_block': {
        4096: {'nop': 1, 'sub    rsp, 0x08': 1, 'mov    rax, qword ds:[rip + 0x0000000000002fd9<12249,absolute=0x0000000000003fe8>]': 1, 'test   rax, rax': 1, 'je     0x0000000000001016<4118>': 1},
        4116: {'call   rax': 1},
        4118: {'add    rsp, 0x08': 1, 'ret': 1},
        4160: {'nop': 1, 'jmp    qword ds:[rip + 0x0000000000002fad<12205,absolute=0x0000000000003ff8>]': 1},
        4176: {'nop': 1, 'jmp    qword ds:[rip + 0x0000000000002f75<12149,absolute=0x0000000000003fd0>]': 1},
        4192: {'nop': 1, 'xor    ebp, ebp': 1, 'mov    r9, rdx': 1, 'pop    rsi': 1, 'mov    rdx, rsp': 1, 'and    rsp, 0xf0<-16>': 1, 'push   rax': 1, 'push   rsp': 1, 'lea    r8, [rip + 0x00000000000002c6<710,absolute=0x0000000000001340>]': 1, 'lea    rcx, [rip + 0x000000000000024f<591,absolute=0x00000000000012d0>]': 1, 'lea    rdi, [rip + 0x00000000000001a7<423,absolute=0x000000000000122f>]': 1, 'call   qword ds:[rip + 0x0000000000002f52<12114,absolute=0x0000000000003fe0>]': 1},
        4238: {'hlt': 1},
        4240: {'lea    rdi, [rip + 0x0000000000002f79<12153,absolute=0x0000000000004010>]': 1, 'lea    rax, [rip + 0x0000000000002f72<12146,absolute=0x0000000000004010>]': 1, 'cmp    rax, rdi': 1, 'je     0x00000000000010b8<4280>': 1},
        4259: {'mov    rax, qword ds:[rip + 0x0000000000002f2e<12078,absolute=0x0000000000003fd8>]': 1, 'test   rax, rax': 1, 'je     0x00000000000010b8<4280>': 1},
        4271: {'jmp    rax': 1},
        4273: {'nop    dword ds:[rax + 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>]': 1},
        4280: {'ret': 1},
        4288: {'lea    rdi, [rip + 0x0000000000002f49<12105,absolute=0x0000000000004010>]': 1, 'lea    rsi, [rip + 0x0000000000002f42<12098,absolute=0x0000000000004010>]': 1, 'sub    rsi, rdi': 1, 'mov    rax, rsi': 1, 'shr    rsi, 0x3f': 1, 'sar    rax, 0x03': 1, 'add    rsi, rax': 1, 'sar    rsi, 0x01': 1, 'je     0x00000000000010f8<4344>': 1},
        4324: {'mov    rax, qword ds:[rip + 0x0000000000002f05<12037,absolute=0x0000000000003ff0>]': 1, 'test   rax, rax': 1, 'je     0x00000000000010f8<4344>': 1},
        4336: {'jmp    rax': 1},
        4338: {'nop    word ds:[rax + rax + 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>]': 1},
        4344: {'ret': 1},
        4352: {'nop': 1, 'cmp    byte ds:[rip + 0x0000000000002f05<12037,absolute=0x0000000000004010>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>': 1, 'jne    0x0000000000001138<4408>': 1},
        4365: {'push   rbp': 1, 'cmp    qword ds:[rip + 0x0000000000002ee2<12002,absolute=0x0000000000003ff8>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>': 1, 'mov    rbp, rsp': 1, 'je     0x0000000000001127<4391>': 1},
        4379: {'mov    rdi, qword ds:[rip + 0x0000000000002ee6<12006,absolute=0x0000000000004008>]': 1, 'call   0x0000000000001040<4160>': 1},
        4391: {'call   0x0000000000001090<4240,(func)deregister_tm_clones>': 1},
        4396: {'mov    byte ds:[rip + 0x0000000000002edd<11997,absolute=0x0000000000004010>], 0x01': 1, 'pop    rbp': 1, 'ret': 1},
        4405: {'nop    dword ds:[rax]': 1},
        4408: {'ret': 1},
        4416: {'nop': 1, 'jmp    0x00000000000010c0<4288,(func)register_tm_clones>': 1},
        4425: {'nop': 1, 'push   rbp': 1, 'mov    rbp, rsp': 1, 'mov    qword ds:[rbp + 0xe8<-24>], rdi': 1, 'mov    eax, esi': 1, 'mov    byte ds:[rbp + 0xe4<-28>], al': 1, 'mov    rax, qword ds:[rbp + 0xe8<-24>]': 1, 'mov    qword ds:[rbp + 0xf8<-8>], rax': 1, 'jmp    0x0000000000001169<4457>': 1},
        4452: {'add    qword ds:[rbp + 0xe8<-24>], 0x01': 1},
        4457: {'mov    rax, qword ds:[rbp + 0xe8<-24>]': 1, 'movzx  eax, byte ds:[rax]': 1, 'test   al, al': 1, 'je     0x0000000000001180<4480>': 1},
        4468: {'mov    rax, qword ds:[rbp + 0xe8<-24>]': 1, 'movzx  eax, byte ds:[rax]': 1, 'cmp    byte ds:[rbp + 0xe4<-28>], al': 1, 'jne    0x0000000000001164<4452>': 1},
        4480: {'mov    rax, qword ds:[rbp + 0xe8<-24>]': 1, 'movzx  eax, byte ds:[rax]': 1, 'test   al, al': 1, 'jne    0x0000000000001191<4497>': 1},
        4491: {'cmp    byte ds:[rbp + 0xe4<-28>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>': 1, 'jne    0x000000000000119b<4507>': 1},
        4497: {'mov    rax, qword ds:[rbp + 0xe8<-24>]': 1, 'sub    rax, qword ds:[rbp + 0xf8<-8>]': 1, 'jmp    0x00000000000011a0<4512>': 1},
        4507: {'mov    eax, 0xffffffff<-1>': 1},
        4512: {'pop    rbp': 1, 'ret': 1},
        4514: {'nop': 1, 'push   rbp': 1, 'mov    rbp, rsp': 1, 'sub    rsp, 0x30': 1, 'mov    qword ds:[rbp + 0xd8<-40>], rdi': 1, 'mov    dword ds:[rbp + 0xd4<-44>], esi': 1, 'mov    rax, qword ds:[rbp + 0xd8<-40>]': 2, 'mov    qword ds:[rbp + 0xf0<-16>], rax': 1, 'mov    esi, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>': 1, 'mov    rdi, rax': 1, 'call   0x0000000000001149<4425,(func)search_char>': 1},
        4558: {'mov    dword ds:[rbp + 0xec<-20>], eax': 1, 'cmp    dword ds:[rbp + 0xec<-20>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>': 1, 'jns    0x0000000000001216<4630>': 1},
        4567: {'mov    eax, 0x00000001': 1, 'jmp    0x000000000000122d<4653>': 1},
        4574: {'mov    rax, qword ds:[rbp + 0xd8<-40>]': 2, 'movzx  eax, byte ds:[rax]': 1, 'movsx  ecx, al': 1, 'mov    eax, dword ds:[rbp + 0xd4<-44>]': 1, 'cdq': 2, 'idiv   ecx': 1, 'mov    eax, edx': 2, 'idiv   dword ds:[rbp + 0xec<-20>]': 1, 'movsxd rdx, eax': 1, 'mov    rax, qword ds:[rbp + 0xf0<-16>]': 1, 'add    rax, rdx': 1, 'mov    qword ds:[rbp + 0xf8<-8>], rax': 1, 'movzx  edx, byte ds:[rax]': 1, 'mov    rax, qword ds:[rbp + 0xf8<-8>]': 1, 'mov    byte ds:[rax], dl': 1, 'add    qword ds:[rbp + 0xd8<-40>], 0x01': 1},
        4630: {'cmp    qword ds:[rbp + 0xf0<-16>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>': 1, 'je     0x0000000000001228<4648>': 1},
        4637: {'mov    rax, qword ds:[rbp + 0xd8<-40>]': 1, 'movzx  eax, byte ds:[rax]': 1, 'test   al, al': 1, 'jne    0x00000000000011de<4574>': 1},
        4648: {'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>': 1},
        4653: {'leave': 1, 'ret': 1},
        4655: {'nop': 1, 'push   rbp': 1, 'mov    rbp, rsp': 1, 'sub    rsp, 0x20': 1, 'mov    dword ds:[rbp + 0xec<-20>], edi': 1, 'mov    qword ds:[rbp + 0xe0<-32>], rsi': 1, 'mov    rax, qword ds:[rbp + 0xe0<-32>]': 1, 'mov    rax, qword ds:[rax + 0x08]': 1, 'mov    qword ds:[rbp + 0xf8<-8>], rax': 1, 'mov    rax, qword ds:[rbp + 0xf8<-8>]': 1, 'mov    esi, 0x0000003a': 1, 'mov    rdi, rax': 1, 'call   0x0000000000001149<4425,(func)search_char>': 1},
        4703: {'mov    dword ds:[rbp + 0xf0<-16>], eax': 1, 'mov    eax, dword ds:[rbp + 0xf0<-16>]': 1, 'mov    esi, eax': 1, 'lea    rdi, [rip + 0x0000000000000d96<3478,absolute=0x0000000000002004>]': 1, 'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>': 1, 'call   0x0000000000001050<4176>': 1},
        4728: {'mov    rax, qword ds:[rbp + 0xf8<-8>]': 1, 'mov    rsi, rax': 1, 'lea    rdi, [rip + 0x0000000000000d8d<3469,absolute=0x0000000000002013>]': 1, 'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>': 1, 'call   0x0000000000001050<4176>': 1},
        4752: {'mov    rax, qword ds:[rbp + 0xf8<-8>]': 1, 'mov    esi, 0x00007ab7<31415>': 1, 'mov    rdi, rax': 1, 'call   0x00000000000011a2<4514,(func)mutate>': 1},
        4769: {'mov    dword ds:[rbp + 0xf4<-12>], eax': 1, 'mov    rax, qword ds:[rbp + 0xf8<-8>]': 1, 'mov    rsi, rax': 1, 'lea    rdi, [rip + 0x0000000000000d6d<3437,absolute=0x000000000000201f>]': 1, 'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>': 1, 'call   0x0000000000001050<4176>': 1},
        4796: {'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>': 1, 'leave': 1, 'ret': 1},
        4816: {'nop': 1, 'push   r15': 1, 'lea    r15, [rip + 0x0000000000002adb<10971,absolute=0x0000000000003db8>]': 1, 'push   r14': 1, 'mov    r14, rdx': 1, 'push   r13': 1, 'mov    r13, rsi': 1, 'push   r12': 1, 'mov    r12d, edi': 1, 'push   rbp': 1, 'lea    rbp, [rip + 0x0000000000002acc<10956,absolute=0x0000000000003dc0>]': 1, 'push   rbx': 1, 'sub    rbp, r15': 1, 'sub    rsp, 0x08': 1, 'call   0x0000000000001000<4096,(func)_init>': 1},
        4865: {'sar    rbp, 0x03': 1, 'je     0x0000000000001326<4902>': 1},
        4871: {'xor    ebx, ebx': 1, 'nop    dword ds:[rax + 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>]': 1},
        4880: {'mov    rdx, r14': 1, 'mov    rsi, r13': 1, 'mov    edi, r12d': 1, 'call   qword ds:[r15 + rbx*0x08]': 1},
        4893: {'add    rbx, 0x01': 1, 'cmp    rbp, rbx': 1, 'jne    0x0000000000001310<4880>': 1},
        4902: {'add    rsp, 0x08': 1, 'pop    rbx': 1, 'pop    rbp': 1, 'pop    r12': 1, 'pop    r13': 1, 'pop    r14': 1, 'pop    r15': 1, 'ret': 1},
        4928: {'nop': 1, 'ret': 1},
        4936: {'nop': 1, 'sub    rsp, 0x08': 1, 'add    rsp, 0x08': 1, 'ret': 1},
      },
      'asm_counts_per_function': {
        4096: {
          'nop': 1,
          'sub    rsp, 0x08': 1,
          'mov    rax, qword ds:[rip + 0x0000000000002fd9<12249,absolute=0x0000000000003fe8>]': 1,
          'test   rax, rax': 1,
          'je     0x0000000000001016<4118>': 1,
          'call   rax': 1,
          'add    rsp, 0x08': 1,
          'ret': 1,
        },
        4160: {
          'nop': 1,
          'jmp    qword ds:[rip + 0x0000000000002fad<12205,absolute=0x0000000000003ff8>]': 1,
        },
        4176: {
          'nop': 1,
          'jmp    qword ds:[rip + 0x0000000000002f75<12149,absolute=0x0000000000003fd0>]': 1,
        },
        4192: {
          'nop': 1,
          'xor    ebp, ebp': 1,
          'mov    r9, rdx': 1,
          'pop    rsi': 1,
          'mov    rdx, rsp': 1,
          'and    rsp, 0xf0<-16>': 1,
          'push   rax': 1,
          'push   rsp': 1,
          'lea    r8, [rip + 0x00000000000002c6<710,absolute=0x0000000000001340>]': 1,
          'lea    rcx, [rip + 0x000000000000024f<591,absolute=0x00000000000012d0>]': 1,
          'lea    rdi, [rip + 0x00000000000001a7<423,absolute=0x000000000000122f>]': 1,
          'call   qword ds:[rip + 0x0000000000002f52<12114,absolute=0x0000000000003fe0>]': 1,
          'hlt': 1,
        },
        4240: {
          'lea    rdi, [rip + 0x0000000000002f79<12153,absolute=0x0000000000004010>]': 1,
          'lea    rax, [rip + 0x0000000000002f72<12146,absolute=0x0000000000004010>]': 1,
          'cmp    rax, rdi': 1,
          'je     0x00000000000010b8<4280>': 2,
          'mov    rax, qword ds:[rip + 0x0000000000002f2e<12078,absolute=0x0000000000003fd8>]': 1,
          'test   rax, rax': 1,
          'jmp    rax': 1,
          'nop    dword ds:[rax + 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>]': 1,
          'ret': 1,
        },
        4288: {
          'lea    rdi, [rip + 0x0000000000002f49<12105,absolute=0x0000000000004010>]': 1,
          'lea    rsi, [rip + 0x0000000000002f42<12098,absolute=0x0000000000004010>]': 1,
          'sub    rsi, rdi': 1,
          'mov    rax, rsi': 1,
          'shr    rsi, 0x3f': 1,
          'sar    rax, 0x03': 1,
          'add    rsi, rax': 1,
          'sar    rsi, 0x01': 1,
          'je     0x00000000000010f8<4344>': 2,
          'mov    rax, qword ds:[rip + 0x0000000000002f05<12037,absolute=0x0000000000003ff0>]': 1,
          'test   rax, rax': 1,
          'jmp    rax': 1,
          'nop    word ds:[rax + rax + 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>]': 1,
          'ret': 1,
        },
        4352: {
          'nop': 1,
          'cmp    byte ds:[rip + 0x0000000000002f05<12037,absolute=0x0000000000004010>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>': 1,
          'jne    0x0000000000001138<4408>': 1,
          'push   rbp': 1,
          'cmp    qword ds:[rip + 0x0000000000002ee2<12002,absolute=0x0000000000003ff8>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>': 1,
          'mov    rbp, rsp': 1,
          'je     0x0000000000001127<4391>': 1,
          'mov    rdi, qword ds:[rip + 0x0000000000002ee6<12006,absolute=0x0000000000004008>]': 1,
          'call   0x0000000000001040<4160>': 1,
          'call   0x0000000000001090<4240,(func)deregister_tm_clones>': 1,
          'mov    byte ds:[rip + 0x0000000000002edd<11997,absolute=0x0000000000004010>], 0x01': 1,
          'pop    rbp': 1,
          'ret': 2,
          'nop    dword ds:[rax]': 1,
        },
        4416: {
          'nop': 1,
          'jmp    0x00000000000010c0<4288,(func)register_tm_clones>': 1,
        },
        4425: {
          'nop': 1,
          'push   rbp': 1,
          'mov    rbp, rsp': 1,
          'mov    qword ds:[rbp + 0xe8<-24>], rdi': 1,
          'mov    eax, esi': 1,
          'mov    byte ds:[rbp + 0xe4<-28>], al': 1,
          'mov    rax, qword ds:[rbp + 0xe8<-24>]': 5,
          'mov    qword ds:[rbp + 0xf8<-8>], rax': 1,
          'jmp    0x0000000000001169<4457>': 1,
          'add    qword ds:[rbp + 0xe8<-24>], 0x01': 1,
          'movzx  eax, byte ds:[rax]': 3,
          'test   al, al': 2,
          'je     0x0000000000001180<4480>': 1,
          'cmp    byte ds:[rbp + 0xe4<-28>], al': 1,
          'jne    0x0000000000001164<4452>': 1,
          'jne    0x0000000000001191<4497>': 1,
          'cmp    byte ds:[rbp + 0xe4<-28>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>': 1,
          'jne    0x000000000000119b<4507>': 1,
          'sub    rax, qword ds:[rbp + 0xf8<-8>]': 1,
          'jmp    0x00000000000011a0<4512>': 1,
          'mov    eax, 0xffffffff<-1>': 1,
          'pop    rbp': 1,
          'ret': 1,
        },
        4514: {
          'nop': 1,
          'push   rbp': 1,
          'mov    rbp, rsp': 1,
          'sub    rsp, 0x30': 1,
          'mov    qword ds:[rbp + 0xd8<-40>], rdi': 1,
          'mov    dword ds:[rbp + 0xd4<-44>], esi': 1,
          'mov    rax, qword ds:[rbp + 0xd8<-40>]': 5,
          'mov    qword ds:[rbp + 0xf0<-16>], rax': 1,
          'mov    esi, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>': 1,
          'mov    rdi, rax': 1,
          'call   0x0000000000001149<4425,(func)search_char>': 1,
          'mov    dword ds:[rbp + 0xec<-20>], eax': 1,
          'cmp    dword ds:[rbp + 0xec<-20>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>': 1,
          'jns    0x0000000000001216<4630>': 1,
          'mov    eax, 0x00000001': 1,
          'jmp    0x000000000000122d<4653>': 1,
          'movzx  eax, byte ds:[rax]': 2,
          'movsx  ecx, al': 1,
          'mov    eax, dword ds:[rbp + 0xd4<-44>]': 1,
          'cdq': 2,
          'idiv   ecx': 1,
          'mov    eax, edx': 2,
          'idiv   dword ds:[rbp + 0xec<-20>]': 1,
          'movsxd rdx, eax': 1,
          'mov    rax, qword ds:[rbp + 0xf0<-16>]': 1,
          'add    rax, rdx': 1,
          'mov    qword ds:[rbp + 0xf8<-8>], rax': 1,
          'movzx  edx, byte ds:[rax]': 1,
          'mov    rax, qword ds:[rbp + 0xf8<-8>]': 1,
          'mov    byte ds:[rax], dl': 1,
          'add    qword ds:[rbp + 0xd8<-40>], 0x01': 1,
          'cmp    qword ds:[rbp + 0xf0<-16>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>': 1,
          'je     0x0000000000001228<4648>': 1,
          'test   al, al': 1,
          'jne    0x00000000000011de<4574>': 1,
          'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>': 1,
          'leave': 1,
          'ret': 1,
        },
        4655: {
          'nop': 1,
          'push   rbp': 1,
          'mov    rbp, rsp': 1,
          'sub    rsp, 0x20': 1,
          'mov    dword ds:[rbp + 0xec<-20>], edi': 1,
          'mov    qword ds:[rbp + 0xe0<-32>], rsi': 1,
          'mov    rax, qword ds:[rbp + 0xe0<-32>]': 1,
          'mov    rax, qword ds:[rax + 0x08]': 1,
          'mov    qword ds:[rbp + 0xf8<-8>], rax': 1,
          'mov    rax, qword ds:[rbp + 0xf8<-8>]': 4,
          'mov    esi, 0x0000003a': 1,
          'mov    rdi, rax': 2,
          'call   0x0000000000001149<4425,(func)search_char>': 1,
          'mov    dword ds:[rbp + 0xf0<-16>], eax': 1,
          'mov    eax, dword ds:[rbp + 0xf0<-16>]': 1,
          'mov    esi, eax': 1,
          'lea    rdi, [rip + 0x0000000000000d96<3478,absolute=0x0000000000002004>]': 1,
          'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>': 4,
          'call   0x0000000000001050<4176>': 3,
          'mov    rsi, rax': 2,
          'lea    rdi, [rip + 0x0000000000000d8d<3469,absolute=0x0000000000002013>]': 1,
          'mov    esi, 0x00007ab7<31415>': 1,
          'call   0x00000000000011a2<4514,(func)mutate>': 1,
          'mov    dword ds:[rbp + 0xf4<-12>], eax': 1,
          'lea    rdi, [rip + 0x0000000000000d6d<3437,absolute=0x000000000000201f>]': 1,
          'leave': 1,
          'ret': 1,
        },
        4816: {
          'nop': 1,
          'push   r15': 1,
          'lea    r15, [rip + 0x0000000000002adb<10971,absolute=0x0000000000003db8>]': 1,
          'push   r14': 1,
          'mov    r14, rdx': 1,
          'push   r13': 1,
          'mov    r13, rsi': 1,
          'push   r12': 1,
          'mov    r12d, edi': 1,
          'push   rbp': 1,
          'lea    rbp, [rip + 0x0000000000002acc<10956,absolute=0x0000000000003dc0>]': 1,
          'push   rbx': 1,
          'sub    rbp, r15': 1,
          'sub    rsp, 0x08': 1,
          'call   0x0000000000001000<4096,(func)_init>': 1,
          'sar    rbp, 0x03': 1,
          'je     0x0000000000001326<4902>': 1,
          'xor    ebx, ebx': 1,
          'nop    dword ds:[rax + 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>]': 1,
          'mov    rdx, r14': 1,
          'mov    rsi, r13': 1,
          'mov    edi, r12d': 1,
          'call   qword ds:[r15 + rbx*0x08]': 1,
          'add    rbx, 0x01': 1,
          'cmp    rbp, rbx': 1,
          'jne    0x0000000000001310<4880>': 1,
          'add    rsp, 0x08': 1,
          'pop    rbx': 1,
          'pop    rbp': 1,
          'pop    r12': 1,
          'pop    r13': 1,
          'pop    r14': 1,
          'pop    r15': 1,
          'ret': 1,
        },
        4928: {
          'nop': 1,
          'ret': 1,
        },
        4936: {
          'nop': 1,
          'sub    rsp, 0x08': 1,
          'add    rsp, 0x08': 1,
          'ret': 1,
        },
      },
      'asm_counts': {
        'nop': 12,
        'sub    rsp, 0x08': 3,
        'mov    rax, qword ds:[rip + 0x0000000000002fd9<12249,absolute=0x0000000000003fe8>]': 1,
        'test   rax, rax': 3,
        'je     0x0000000000001016<4118>': 1,
        'call   rax': 1,
        'add    rsp, 0x08': 3,
        'ret': 11,
        'jmp    qword ds:[rip + 0x0000000000002fad<12205,absolute=0x0000000000003ff8>]': 1,
        'jmp    qword ds:[rip + 0x0000000000002f75<12149,absolute=0x0000000000003fd0>]': 1,
        'xor    ebp, ebp': 1,
        'mov    r9, rdx': 1,
        'pop    rsi': 1,
        'mov    rdx, rsp': 1,
        'and    rsp, 0xf0<-16>': 1,
        'push   rax': 1,
        'push   rsp': 1,
        'lea    r8, [rip + 0x00000000000002c6<710,absolute=0x0000000000001340>]': 1,
        'lea    rcx, [rip + 0x000000000000024f<591,absolute=0x00000000000012d0>]': 1,
        'lea    rdi, [rip + 0x00000000000001a7<423,absolute=0x000000000000122f>]': 1,
        'call   qword ds:[rip + 0x0000000000002f52<12114,absolute=0x0000000000003fe0>]': 1,
        'hlt': 1,
        'lea    rdi, [rip + 0x0000000000002f79<12153,absolute=0x0000000000004010>]': 1,
        'lea    rax, [rip + 0x0000000000002f72<12146,absolute=0x0000000000004010>]': 1,
        'cmp    rax, rdi': 1,
        'je     0x00000000000010b8<4280>': 2,
        'mov    rax, qword ds:[rip + 0x0000000000002f2e<12078,absolute=0x0000000000003fd8>]': 1,
        'jmp    rax': 2,
        'nop    dword ds:[rax + 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>]': 2,
        'lea    rdi, [rip + 0x0000000000002f49<12105,absolute=0x0000000000004010>]': 1,
        'lea    rsi, [rip + 0x0000000000002f42<12098,absolute=0x0000000000004010>]': 1,
        'sub    rsi, rdi': 1,
        'mov    rax, rsi': 1,
        'shr    rsi, 0x3f': 1,
        'sar    rax, 0x03': 1,
        'add    rsi, rax': 1,
        'sar    rsi, 0x01': 1,
        'je     0x00000000000010f8<4344>': 2,
        'mov    rax, qword ds:[rip + 0x0000000000002f05<12037,absolute=0x0000000000003ff0>]': 1,
        'nop    word ds:[rax + rax + 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>]': 1,
        'cmp    byte ds:[rip + 0x0000000000002f05<12037,absolute=0x0000000000004010>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>': 1,
        'jne    0x0000000000001138<4408>': 1,
        'push   rbp': 5,
        'cmp    qword ds:[rip + 0x0000000000002ee2<12002,absolute=0x0000000000003ff8>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>': 1,
        'mov    rbp, rsp': 4,
        'je     0x0000000000001127<4391>': 1,
        'mov    rdi, qword ds:[rip + 0x0000000000002ee6<12006,absolute=0x0000000000004008>]': 1,
        'call   0x0000000000001040<4160>': 1,
        'call   0x0000000000001090<4240,(func)deregister_tm_clones>': 1,
        'mov    byte ds:[rip + 0x0000000000002edd<11997,absolute=0x0000000000004010>], 0x01': 1,
        'pop    rbp': 3,
        'nop    dword ds:[rax]': 1,
        'jmp    0x00000000000010c0<4288,(func)register_tm_clones>': 1,
        'mov    qword ds:[rbp + 0xe8<-24>], rdi': 1,
        'mov    eax, esi': 1,
        'mov    byte ds:[rbp + 0xe4<-28>], al': 1,
        'mov    rax, qword ds:[rbp + 0xe8<-24>]': 5,
        'mov    qword ds:[rbp + 0xf8<-8>], rax': 3,
        'jmp    0x0000000000001169<4457>': 1,
        'add    qword ds:[rbp + 0xe8<-24>], 0x01': 1,
        'movzx  eax, byte ds:[rax]': 5,
        'test   al, al': 3,
        'je     0x0000000000001180<4480>': 1,
        'cmp    byte ds:[rbp + 0xe4<-28>], al': 1,
        'jne    0x0000000000001164<4452>': 1,
        'jne    0x0000000000001191<4497>': 1,
        'cmp    byte ds:[rbp + 0xe4<-28>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>': 1,
        'jne    0x000000000000119b<4507>': 1,
        'sub    rax, qword ds:[rbp + 0xf8<-8>]': 1,
        'jmp    0x00000000000011a0<4512>': 1,
        'mov    eax, 0xffffffff<-1>': 1,
        'sub    rsp, 0x30': 1,
        'mov    qword ds:[rbp + 0xd8<-40>], rdi': 1,
        'mov    dword ds:[rbp + 0xd4<-44>], esi': 1,
        'mov    rax, qword ds:[rbp + 0xd8<-40>]': 5,
        'mov    qword ds:[rbp + 0xf0<-16>], rax': 1,
        'mov    esi, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>': 1,
        'mov    rdi, rax': 3,
        'call   0x0000000000001149<4425,(func)search_char>': 2,
        'mov    dword ds:[rbp + 0xec<-20>], eax': 1,
        'cmp    dword ds:[rbp + 0xec<-20>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>': 1,
        'jns    0x0000000000001216<4630>': 1,
        'mov    eax, 0x00000001': 1,
        'jmp    0x000000000000122d<4653>': 1,
        'movsx  ecx, al': 1,
        'mov    eax, dword ds:[rbp + 0xd4<-44>]': 1,
        'cdq': 2,
        'idiv   ecx': 1,
        'mov    eax, edx': 2,
        'idiv   dword ds:[rbp + 0xec<-20>]': 1,
        'movsxd rdx, eax': 1,
        'mov    rax, qword ds:[rbp + 0xf0<-16>]': 1,
        'add    rax, rdx': 1,
        'movzx  edx, byte ds:[rax]': 1,
        'mov    rax, qword ds:[rbp + 0xf8<-8>]': 5,
        'mov    byte ds:[rax], dl': 1,
        'add    qword ds:[rbp + 0xd8<-40>], 0x01': 1,
        'cmp    qword ds:[rbp + 0xf0<-16>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>': 1,
        'je     0x0000000000001228<4648>': 1,
        'jne    0x00000000000011de<4574>': 1,
        'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>': 5,
        'leave': 2,
        'sub    rsp, 0x20': 1,
        'mov    dword ds:[rbp + 0xec<-20>], edi': 1,
        'mov    qword ds:[rbp + 0xe0<-32>], rsi': 1,
        'mov    rax, qword ds:[rbp + 0xe0<-32>]': 1,
        'mov    rax, qword ds:[rax + 0x08]': 1,
        'mov    esi, 0x0000003a': 1,
        'mov    dword ds:[rbp + 0xf0<-16>], eax': 1,
        'mov    eax, dword ds:[rbp + 0xf0<-16>]': 1,
        'mov    esi, eax': 1,
        'lea    rdi, [rip + 0x0000000000000d96<3478,absolute=0x0000000000002004>]': 1,
        'call   0x0000000000001050<4176>': 3,
        'mov    rsi, rax': 2,
        'lea    rdi, [rip + 0x0000000000000d8d<3469,absolute=0x0000000000002013>]': 1,
        'mov    esi, 0x00007ab7<31415>': 1,
        'call   0x00000000000011a2<4514,(func)mutate>': 1,
        'mov    dword ds:[rbp + 0xf4<-12>], eax': 1,
        'lea    rdi, [rip + 0x0000000000000d6d<3437,absolute=0x000000000000201f>]': 1,
        'push   r15': 1,
        'lea    r15, [rip + 0x0000000000002adb<10971,absolute=0x0000000000003db8>]': 1,
        'push   r14': 1,
        'mov    r14, rdx': 1,
        'push   r13': 1,
        'mov    r13, rsi': 1,
        'push   r12': 1,
        'mov    r12d, edi': 1,
        'lea    rbp, [rip + 0x0000000000002acc<10956,absolute=0x0000000000003dc0>]': 1,
        'push   rbx': 1,
        'sub    rbp, r15': 1,
        'call   0x0000000000001000<4096,(func)_init>': 1,
        'sar    rbp, 0x03': 1,
        'je     0x0000000000001326<4902>': 1,
        'xor    ebx, ebx': 1,
        'mov    rdx, r14': 1,
        'mov    rsi, r13': 1,
        'mov    edi, r12d': 1,
        'call   qword ds:[r15 + rbx*0x08]': 1,
        'add    rbx, 0x01': 1,
        'cmp    rbp, rbx': 1,
        'jne    0x0000000000001310<4880>': 1,
        'pop    rbx': 1,
        'pop    r12': 1,
        'pop    r13': 1,
        'pop    r14': 1,
        'pop    r15': 1,
      },
    }

    # Adding functions to the cfg
    if build_level in ['cfg']:
        __auto_cfg.add_function(*__auto_functions.values())
    else:
        __auto_cfg.functions = list(__auto_functions.values())
        __auto_cfg.blocks = list(__auto_blocks.values())

    return {
        'blocks': __auto_blocks,
        'file': os.path.basename(__file__),
        'inputs': [MANUAL_ROSE_GV_STR, MANUAL_ROSE_TXT_STR],
        'cfg': __auto_cfg,
        'functions': __auto_functions,
        'expected': expected,
    }


MANUAL_ROSE_TXT_STR = """
function 0x00001000 "_init"
  B1 function entry point:
    function call edge from function 0x000012d0 "__libc_csu_init"
      0x00001000: nop    
      0x00001004: sub    rsp, 0x08
      0x00001008: mov    rax, qword ds:[rip + 0x0000000000002fd9<12249,absolute=0x0000000000003fe8>]
      0x0000100f: test   rax, rax
      0x00001012: je     0x0000000000001016<4118>
    normal edge to B2 basic block 0x00001016
    normal edge to B3 basic block 0x00001014
  B3 basic block 0x00001014:
    normal edge from B1 function entry point
      0x00001014: call   rax
    block is a function call
    call return edge to B2 basic block 0x00001016
    function call edge to indeterminate
  B2 basic block 0x00001016:
    normal edge from B1 function entry point
    call return edge from B3 basic block 0x00001014
      0x00001016: add    rsp, 0x08
      0x0000101a: ret    
    block is a function return
    function return edge to indeterminate
function 0x00001040
  B1 function entry point:
    function call edge from basic block 0x0000111b
      0x00001040: nop    
      0x00001044: jmp    qword ds:[rip + 0x0000000000002fad<12205,absolute=0x0000000000003ff8>]
    normal edge to indeterminate
function 0x00001050 "printf@plt"
  B1 function entry point:
    function call edge from basic block 0x0000125f
    function call edge from basic block 0x00001278
    function call edge from basic block 0x000012a1
      0x00001050: nop    
      0x00001054: jmp    qword ds:[rip + 0x0000000000002f75<12149,absolute=0x0000000000003fd0>]
    normal edge to indeterminate
function 0x00001060 "_start"
  B1 function entry point:
      0x00001060: nop    
      0x00001064: xor    ebp, ebp
      0x00001066: mov    r9, rdx
      0x00001069: pop    rsi
      0x0000106a: mov    rdx, rsp
      0x0000106d: and    rsp, 0xf0<-16>
      0x00001071: push   rax
      0x00001072: push   rsp
      0x00001073: lea    r8, [rip + 0x00000000000002c6<710,absolute=0x0000000000001340>]
      0x0000107a: lea    rcx, [rip + 0x000000000000024f<591,absolute=0x00000000000012d0>]
      0x00001081: lea    rdi, [rip + 0x00000000000001a7<423,absolute=0x000000000000122f>]
      0x00001088: call   qword ds:[rip + 0x0000000000002f52<12114,absolute=0x0000000000003fe0>]
    block is a function call
    call return edge to B2 basic block 0x0000108e
    function call edge to indeterminate
  B2 basic block 0x0000108e:
    call return edge from B1 function entry point
      0x0000108e: hlt    
function 0x00001090 "deregister_tm_clones"
  B1 function entry point:
    function call edge from basic block 0x00001127
      0x00001090: lea    rdi, [rip + 0x0000000000002f79<12153,absolute=0x0000000000004010>]
      0x00001097: lea    rax, [rip + 0x0000000000002f72<12146,absolute=0x0000000000004010>]
      0x0000109e: cmp    rax, rdi
      0x000010a1: je     0x00000000000010b8<4280>
    normal edge to B2 basic block 0x000010b8
    normal edge to B3 basic block 0x000010a3
  B3 basic block 0x000010a3:
    normal edge from B1 function entry point
      0x000010a3: mov    rax, qword ds:[rip + 0x0000000000002f2e<12078,absolute=0x0000000000003fd8>]
      0x000010aa: test   rax, rax
      0x000010ad: je     0x00000000000010b8<4280>
    normal edge to B2 basic block 0x000010b8
    normal edge to B4 basic block 0x000010af
  B4 basic block 0x000010af:
    normal edge from B3 basic block 0x000010a3
      0x000010af: jmp    rax
    normal edge to indeterminate
  B5 basic block 0x000010b1:
      0x000010b1: nop    dword ds:[rax + 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>]
    normal edge to B2 basic block 0x000010b8
  B2 basic block 0x000010b8:
    normal edge from B1 function entry point
    normal edge from B3 basic block 0x000010a3
    normal edge from B5 basic block 0x000010b1
      0x000010b8: ret    
    block is a function return
    function return edge to indeterminate
function 0x000010c0 "register_tm_clones"
  B1 function entry point:
    normal edge from function 0x00001140 "frame_dummy"
      0x000010c0: lea    rdi, [rip + 0x0000000000002f49<12105,absolute=0x0000000000004010>]
      0x000010c7: lea    rsi, [rip + 0x0000000000002f42<12098,absolute=0x0000000000004010>]
      0x000010ce: sub    rsi, rdi
      0x000010d1: mov    rax, rsi
      0x000010d4: shr    rsi, 0x3f
      0x000010d8: sar    rax, 0x03
      0x000010dc: add    rsi, rax
      0x000010df: sar    rsi, 0x01
      0x000010e2: je     0x00000000000010f8<4344>
    normal edge to B2 basic block 0x000010f8
    normal edge to B3 basic block 0x000010e4
  B3 basic block 0x000010e4:
    normal edge from B1 function entry point
      0x000010e4: mov    rax, qword ds:[rip + 0x0000000000002f05<12037,absolute=0x0000000000003ff0>]
      0x000010eb: test   rax, rax
      0x000010ee: je     0x00000000000010f8<4344>
    normal edge to B2 basic block 0x000010f8
    normal edge to B4 basic block 0x000010f0
  B4 basic block 0x000010f0:
    normal edge from B3 basic block 0x000010e4
      0x000010f0: jmp    rax
    normal edge to indeterminate
  B5 basic block 0x000010f2:
      0x000010f2: nop    word ds:[rax + rax + 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>]
    normal edge to B2 basic block 0x000010f8
  B2 basic block 0x000010f8:
    normal edge from B1 function entry point
    normal edge from B3 basic block 0x000010e4
    normal edge from B5 basic block 0x000010f2
      0x000010f8: ret    
    block is a function return
    function return edge to indeterminate
function 0x00001100 "__do_global_dtors_aux"
  B1 function entry point:
      0x00001100: nop    
      0x00001104: cmp    byte ds:[rip + 0x0000000000002f05<12037,absolute=0x0000000000004010>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>
      0x0000110b: jne    0x0000000000001138<4408>
    normal edge to B2 basic block 0x00001138
    normal edge to B3 basic block 0x0000110d
  B3 basic block 0x0000110d:
    normal edge from B1 function entry point
      0x0000110d: push   rbp
      0x0000110e: cmp    qword ds:[rip + 0x0000000000002ee2<12002,absolute=0x0000000000003ff8>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>
      0x00001116: mov    rbp, rsp
      0x00001119: je     0x0000000000001127<4391>
    normal edge to B4 basic block 0x00001127
    normal edge to B5 basic block 0x0000111b
  B5 basic block 0x0000111b:
    normal edge from B3 basic block 0x0000110d
      0x0000111b: mov    rdi, qword ds:[rip + 0x0000000000002ee6<12006,absolute=0x0000000000004008>]
      0x00001122: call   0x0000000000001040<4160>
    block is a function call
    call return edge to B4 basic block 0x00001127
    function call edge to function 0x00001040
  B4 basic block 0x00001127:
    normal edge from B3 basic block 0x0000110d
    call return edge from B5 basic block 0x0000111b
      0x00001127: call   0x0000000000001090<4240,(func)deregister_tm_clones>
    block is a function call
    call return edge to B6 basic block 0x0000112c
    function call edge to function 0x00001090 "deregister_tm_clones"
  B6 basic block 0x0000112c:
    call return edge from B4 basic block 0x00001127
      0x0000112c: mov    byte ds:[rip + 0x0000000000002edd<11997,absolute=0x0000000000004010>], 0x01
      0x00001133: pop    rbp
      0x00001134: ret    
    block is a function return
    function return edge to indeterminate
  B7 basic block 0x00001135:
      0x00001135: nop    dword ds:[rax]
    normal edge to B2 basic block 0x00001138
  B2 basic block 0x00001138:
    normal edge from B1 function entry point
    normal edge from B7 basic block 0x00001135
      0x00001138: ret    
    block is a function return
    function return edge to indeterminate
function 0x00001140 "frame_dummy"
  B1 function entry point:
      0x00001140: nop    
      0x00001144: jmp    0x00000000000010c0<4288,(func)register_tm_clones>
    normal edge to function 0x000010c0 "register_tm_clones"
function 0x00001149 "search_char"
  B1 function entry point:
    function call edge from function 0x000011a2 "mutate"
    function call edge from function 0x0000122f "main"
      0x00001149: nop    
      0x0000114d: push   rbp
      0x0000114e: mov    rbp, rsp
      0x00001151: mov    qword ds:[rbp + 0xe8<-24>], rdi
      0x00001155: mov    eax, esi
      0x00001157: mov    byte ds:[rbp + 0xe4<-28>], al
      0x0000115a: mov    rax, qword ds:[rbp + 0xe8<-24>]
      0x0000115e: mov    qword ds:[rbp + 0xf8<-8>], rax
      0x00001162: jmp    0x0000000000001169<4457>
    normal edge to B2 basic block 0x00001169
  B3 basic block 0x00001164:
    normal edge from B4 basic block 0x00001174
      0x00001164: add    qword ds:[rbp + 0xe8<-24>], 0x01
    normal edge to B2 basic block 0x00001169
  B2 basic block 0x00001169:
    normal edge from B1 function entry point
    normal edge from B3 basic block 0x00001164
      0x00001169: mov    rax, qword ds:[rbp + 0xe8<-24>]
      0x0000116d: movzx  eax, byte ds:[rax]
      0x00001170: test   al, al
      0x00001172: je     0x0000000000001180<4480>
    normal edge to B5 basic block 0x00001180
    normal edge to B4 basic block 0x00001174
  B4 basic block 0x00001174:
    normal edge from B2 basic block 0x00001169
      0x00001174: mov    rax, qword ds:[rbp + 0xe8<-24>]
      0x00001178: movzx  eax, byte ds:[rax]
      0x0000117b: cmp    byte ds:[rbp + 0xe4<-28>], al
      0x0000117e: jne    0x0000000000001164<4452>
    normal edge to B5 basic block 0x00001180
    normal edge to B3 basic block 0x00001164
  B5 basic block 0x00001180:
    normal edge from B2 basic block 0x00001169
    normal edge from B4 basic block 0x00001174
      0x00001180: mov    rax, qword ds:[rbp + 0xe8<-24>]
      0x00001184: movzx  eax, byte ds:[rax]
      0x00001187: test   al, al
      0x00001189: jne    0x0000000000001191<4497>
    normal edge to B6 basic block 0x00001191
    normal edge to B7 basic block 0x0000118b
  B7 basic block 0x0000118b:
    normal edge from B5 basic block 0x00001180
      0x0000118b: cmp    byte ds:[rbp + 0xe4<-28>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>
      0x0000118f: jne    0x000000000000119b<4507>
    normal edge to B8 basic block 0x0000119b
    normal edge to B6 basic block 0x00001191
  B6 basic block 0x00001191:
    normal edge from B5 basic block 0x00001180
    normal edge from B7 basic block 0x0000118b
      0x00001191: mov    rax, qword ds:[rbp + 0xe8<-24>]
      0x00001195: sub    rax, qword ds:[rbp + 0xf8<-8>]
      0x00001199: jmp    0x00000000000011a0<4512>
    normal edge to B9 basic block 0x000011a0
  B8 basic block 0x0000119b:
    normal edge from B7 basic block 0x0000118b
      0x0000119b: mov    eax, 0xffffffff<-1>
    normal edge to B9 basic block 0x000011a0
  B9 basic block 0x000011a0:
    normal edge from B6 basic block 0x00001191
    normal edge from B8 basic block 0x0000119b
      0x000011a0: pop    rbp
      0x000011a1: ret    
    block is a function return
    function return edge to indeterminate
function 0x000011a2 "mutate"
  B1 function entry point:
    function call edge from basic block 0x00001290
      0x000011a2: nop    
      0x000011a6: push   rbp
      0x000011a7: mov    rbp, rsp
      0x000011aa: sub    rsp, 0x30
      0x000011ae: mov    qword ds:[rbp + 0xd8<-40>], rdi
      0x000011b2: mov    dword ds:[rbp + 0xd4<-44>], esi
      0x000011b5: mov    rax, qword ds:[rbp + 0xd8<-40>]
      0x000011b9: mov    qword ds:[rbp + 0xf0<-16>], rax
      0x000011bd: mov    rax, qword ds:[rbp + 0xd8<-40>]
      0x000011c1: mov    esi, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>
      0x000011c6: mov    rdi, rax
      0x000011c9: call   0x0000000000001149<4425,(func)search_char>
    block is a function call
    function call edge to function 0x00001149 "search_char"
    call return edge to B2 basic block 0x000011ce
  B2 basic block 0x000011ce:
    call return edge from B1 function entry point
      0x000011ce: mov    dword ds:[rbp + 0xec<-20>], eax
      0x000011d1: cmp    dword ds:[rbp + 0xec<-20>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>
      0x000011d5: jns    0x0000000000001216<4630>
    normal edge to B3 basic block 0x000011d7
    normal edge to B4 basic block 0x00001216
  B3 basic block 0x000011d7:
    normal edge from B2 basic block 0x000011ce
      0x000011d7: mov    eax, 0x00000001
      0x000011dc: jmp    0x000000000000122d<4653>
    normal edge to B5 basic block 0x0000122d
  B6 basic block 0x000011de:
    normal edge from B7 basic block 0x0000121d
      0x000011de: mov    rax, qword ds:[rbp + 0xd8<-40>]
      0x000011e2: movzx  eax, byte ds:[rax]
      0x000011e5: movsx  ecx, al
      0x000011e8: mov    eax, dword ds:[rbp + 0xd4<-44>]
      0x000011eb: cdq    
      0x000011ec: idiv   ecx
      0x000011ee: mov    eax, edx
      0x000011f0: cdq    
      0x000011f1: idiv   dword ds:[rbp + 0xec<-20>]
      0x000011f4: mov    eax, edx
      0x000011f6: movsxd rdx, eax
      0x000011f9: mov    rax, qword ds:[rbp + 0xf0<-16>]
      0x000011fd: add    rax, rdx
      0x00001200: mov    qword ds:[rbp + 0xf8<-8>], rax
      0x00001204: mov    rax, qword ds:[rbp + 0xd8<-40>]
      0x00001208: movzx  edx, byte ds:[rax]
      0x0000120b: mov    rax, qword ds:[rbp + 0xf8<-8>]
      0x0000120f: mov    byte ds:[rax], dl
      0x00001211: add    qword ds:[rbp + 0xd8<-40>], 0x01
    normal edge to B4 basic block 0x00001216
  B4 basic block 0x00001216:
    normal edge from B2 basic block 0x000011ce
    normal edge from B6 basic block 0x000011de
      0x00001216: cmp    qword ds:[rbp + 0xf0<-16>], 0x00<(func)__cxa_finalize@@GLIBC_2.2.5>
      0x0000121b: je     0x0000000000001228<4648>
    normal edge to B8 basic block 0x00001228
    normal edge to B7 basic block 0x0000121d
  B7 basic block 0x0000121d:
    normal edge from B4 basic block 0x00001216
      0x0000121d: mov    rax, qword ds:[rbp + 0xd8<-40>]
      0x00001221: movzx  eax, byte ds:[rax]
      0x00001224: test   al, al
      0x00001226: jne    0x00000000000011de<4574>
    normal edge to B8 basic block 0x00001228
    normal edge to B6 basic block 0x000011de
  B8 basic block 0x00001228:
    normal edge from B4 basic block 0x00001216
    normal edge from B7 basic block 0x0000121d
      0x00001228: mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>
    normal edge to B5 basic block 0x0000122d
  B5 basic block 0x0000122d:
    normal edge from B8 basic block 0x00001228
    normal edge from B3 basic block 0x000011d7
      0x0000122d: leave  
      0x0000122e: ret    
    block is a function return
    function return edge to indeterminate
function 0x0000122f "main"
  B1 function entry point:
      0x0000122f: nop    
      0x00001233: push   rbp
      0x00001234: mov    rbp, rsp
      0x00001237: sub    rsp, 0x20
      0x0000123b: mov    dword ds:[rbp + 0xec<-20>], edi
      0x0000123e: mov    qword ds:[rbp + 0xe0<-32>], rsi
      0x00001242: mov    rax, qword ds:[rbp + 0xe0<-32>]
      0x00001246: mov    rax, qword ds:[rax + 0x08]
      0x0000124a: mov    qword ds:[rbp + 0xf8<-8>], rax
      0x0000124e: mov    rax, qword ds:[rbp + 0xf8<-8>]
      0x00001252: mov    esi, 0x0000003a
      0x00001257: mov    rdi, rax
      0x0000125a: call   0x0000000000001149<4425,(func)search_char>
    block is a function call
    call return edge to B2 basic block 0x0000125f
    function call edge to function 0x00001149 "search_char"
  B2 basic block 0x0000125f:
    call return edge from B1 function entry point
      0x0000125f: mov    dword ds:[rbp + 0xf0<-16>], eax
      0x00001262: mov    eax, dword ds:[rbp + 0xf0<-16>]
      0x00001265: mov    esi, eax
      0x00001267: lea    rdi, [rip + 0x0000000000000d96<3478,absolute=0x0000000000002004>]
      0x0000126e: mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>
      0x00001273: call   0x0000000000001050<4176>
    block is a function call
    call return edge to B3 basic block 0x00001278
    function call edge to function 0x00001050 "printf@plt"
  B3 basic block 0x00001278:
    call return edge from B2 basic block 0x0000125f
      0x00001278: mov    rax, qword ds:[rbp + 0xf8<-8>]
      0x0000127c: mov    rsi, rax
      0x0000127f: lea    rdi, [rip + 0x0000000000000d8d<3469,absolute=0x0000000000002013>]
      0x00001286: mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>
      0x0000128b: call   0x0000000000001050<4176>
    block is a function call
    call return edge to B4 basic block 0x00001290
    function call edge to function 0x00001050 "printf@plt"
  B4 basic block 0x00001290:
    call return edge from B3 basic block 0x00001278
      0x00001290: mov    rax, qword ds:[rbp + 0xf8<-8>]
      0x00001294: mov    esi, 0x00007ab7<31415>
      0x00001299: mov    rdi, rax
      0x0000129c: call   0x00000000000011a2<4514,(func)mutate>
    block is a function call
    call return edge to B5 basic block 0x000012a1
    function call edge to function 0x000011a2 "mutate"
  B5 basic block 0x000012a1:
    call return edge from B4 basic block 0x00001290
      0x000012a1: mov    dword ds:[rbp + 0xf4<-12>], eax
      0x000012a4: mov    rax, qword ds:[rbp + 0xf8<-8>]
      0x000012a8: mov    rsi, rax
      0x000012ab: lea    rdi, [rip + 0x0000000000000d6d<3437,absolute=0x000000000000201f>]
      0x000012b2: mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>
      0x000012b7: call   0x0000000000001050<4176>
    block is a function call
    call return edge to B6 basic block 0x000012bc
    function call edge to function 0x00001050 "printf@plt"
  B6 basic block 0x000012bc:
    call return edge from B5 basic block 0x000012a1
      0x000012bc: mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>
      0x000012c1: leave  
      0x000012c2: ret    
    block is a function return
    function return edge to indeterminate
function 0x000012d0 "__libc_csu_init"
  B1 function entry point:
      0x000012d0: nop    
      0x000012d4: push   r15
      0x000012d6: lea    r15, [rip + 0x0000000000002adb<10971,absolute=0x0000000000003db8>]
      0x000012dd: push   r14
      0x000012df: mov    r14, rdx
      0x000012e2: push   r13
      0x000012e4: mov    r13, rsi
      0x000012e7: push   r12
      0x000012e9: mov    r12d, edi
      0x000012ec: push   rbp
      0x000012ed: lea    rbp, [rip + 0x0000000000002acc<10956,absolute=0x0000000000003dc0>]
      0x000012f4: push   rbx
      0x000012f5: sub    rbp, r15
      0x000012f8: sub    rsp, 0x08
      0x000012fc: call   0x0000000000001000<4096,(func)_init>
    block is a function call
    function call edge to function 0x00001000 "_init"
    call return edge to B2 basic block 0x00001301
  B2 basic block 0x00001301:
    call return edge from B1 function entry point
      0x00001301: sar    rbp, 0x03
      0x00001305: je     0x0000000000001326<4902>
    normal edge to B3 basic block 0x00001326
    normal edge to B4 basic block 0x00001307
  B4 basic block 0x00001307:
    normal edge from B2 basic block 0x00001301
      0x00001307: xor    ebx, ebx
      0x00001309: nop    dword ds:[rax + 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>]
    normal edge to B5 basic block 0x00001310
  B5 basic block 0x00001310:
    normal edge from B6 basic block 0x0000131d
    normal edge from B4 basic block 0x00001307
      0x00001310: mov    rdx, r14
      0x00001313: mov    rsi, r13
      0x00001316: mov    edi, r12d
      0x00001319: call   qword ds:[r15 + rbx*0x08]
    block is a function call
    function call edge to indeterminate
    call return edge to B6 basic block 0x0000131d
  B6 basic block 0x0000131d:
    call return edge from B5 basic block 0x00001310
      0x0000131d: add    rbx, 0x01
      0x00001321: cmp    rbp, rbx
      0x00001324: jne    0x0000000000001310<4880>
    normal edge to B5 basic block 0x00001310
    normal edge to B3 basic block 0x00001326
  B3 basic block 0x00001326:
    normal edge from B2 basic block 0x00001301
    normal edge from B6 basic block 0x0000131d
      0x00001326: add    rsp, 0x08
      0x0000132a: pop    rbx
      0x0000132b: pop    rbp
      0x0000132c: pop    r12
      0x0000132e: pop    r13
      0x00001330: pop    r14
      0x00001332: pop    r15
      0x00001334: ret    
    block is a function return
    function return edge to indeterminate
function 0x00001340 "__libc_csu_fini"
  B1 function entry point:
      0x00001340: nop    
      0x00001344: ret    
    block is a function return
    function return edge to indeterminate
function 0x00001348 "_fini"
  B1 function entry point:
      0x00001348: nop    
      0x0000134c: sub    rsp, 0x08
      0x00001350: add    rsp, 0x08
      0x00001354: ret    
    block is a function return
    function return edge to indeterminate
"""


MANUAL_ROSE_GV_STR = """
digraph CFG {
 graph [ overlap=scale ];
 node  [  ];
 edge  [  ];

subgraph cluster_0x00001000 { label="function 0x00001000 \\"_init\\"" fillcolor="#f2f2f2" href="0x00001000" style=filled
V_0x00001000 [ label=<00001000  ?? nop    <br align="left"/>00001004  ?? sub    rsp, 0x08<br align="left"/>00001008  ?? mov    rax, qword ds:[rip + 0x0000000000002fd9&lt;12249,absolute=0x0000000000003fe8&gt;]<br align="left"/>0000100f  ?? test   rax, rax<br align="left"/>00001012  ?? je     0x0000000000001016&lt;4118&gt;<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x00001000" shape=box style=filled ];
V_0x00001014 [ label=<00001014  ?? call   rax<br align="left"/>> fontname=Courier href="0x00001014" shape=box ];
V_0x00001016 [ label=<00001016  ?? add    rsp, 0x08<br align="left"/>0000101a  ?? ret    <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x00001016" shape=box style=filled ];
}

subgraph cluster_0x00001040 { label="function 0x00001040" fillcolor="#f2f2f2" href="0x00001040" style=filled
V_0x00001040 [ label=<00001040  ?? nop    <br align="left"/>00001044  ?? jmp    qword ds:[rip + 0x0000000000002fad&lt;12205,absolute=0x0000000000003ff8&gt;]<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x00001040" shape=box style=filled ];
}

subgraph cluster_0x00001050 { label="function 0x00001050 \\"printf@plt\\"" fillcolor="#f2f2f2" href="0x00001050" style=filled
V_0x00001050 [ label=<00001050  ?? nop    <br align="left"/>00001054  ?? jmp    qword ds:[rip + 0x0000000000002f75&lt;12149,absolute=0x0000000000003fd0&gt;]<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x00001050" shape=box style=filled ];
}

subgraph cluster_0x00001060 { label="function 0x00001060 \\"_start\\"" fillcolor="#f2f2f2" href="0x00001060" style=filled
V_0x00001060 [ label=<00001060  ?? nop    <br align="left"/>00001064  ?? xor    ebp, ebp<br align="left"/>00001066  ?? mov    r9, rdx<br align="left"/>00001069  ?? pop    rsi<br align="left"/>0000106a  ?? mov    rdx, rsp<br align="left"/>0000106d  ?? and    rsp, 0xf0&lt;-16&gt;<br align="left"/>00001071  ?? push   rax<br align="left"/>00001072  ?? push   rsp<br align="left"/>00001073  ?? lea    r8, [rip + 0x00000000000002c6&lt;710,absolute=0x0000000000001340&gt;]<br align="left"/>0000107a  ?? lea    rcx, [rip + 0x000000000000024f&lt;591,absolute=0x00000000000012d0&gt;]<br align="left"/>00001081  ?? lea    rdi, [rip + 0x00000000000001a7&lt;423,absolute=0x000000000000122f&gt;]<br align="left"/>00001088  ?? call   qword ds:[rip + 0x0000000000002f52&lt;12114,absolute=0x0000000000003fe0&gt;]<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x00001060" shape=box style=filled ];
V_0x0000108e [ label=<0000108e  ?? hlt    <br align="left"/>> fontname=Courier href="0x0000108e" shape=box ];
}

subgraph cluster_0x00001090 { label="function 0x00001090 \\"deregister_tm_clones\\"" fillcolor="#f2f2f2" href="0x00001090" style=filled
V_0x00001090 [ label=<00001090  ?? lea    rdi, [rip + 0x0000000000002f79&lt;12153,absolute=0x0000000000004010&gt;]<br align="left"/>00001097  ?? lea    rax, [rip + 0x0000000000002f72&lt;12146,absolute=0x0000000000004010&gt;]<br align="left"/>0000109e  ?? cmp    rax, rdi<br align="left"/>000010a1  ?? je     0x00000000000010b8&lt;4280&gt;<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x00001090" shape=box style=filled ];
V_0x000010a3 [ label=<000010a3  ?? mov    rax, qword ds:[rip + 0x0000000000002f2e&lt;12078,absolute=0x0000000000003fd8&gt;]<br align="left"/>000010aa  ?? test   rax, rax<br align="left"/>000010ad  ?? je     0x00000000000010b8&lt;4280&gt;<br align="left"/>> fontname=Courier href="0x000010a3" shape=box ];
V_0x000010b8 [ label=<000010b8  ?? ret    <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x000010b8" shape=box style=filled ];
V_0x000010af [ label=<000010af  ?? jmp    rax<br align="left"/>> fontname=Courier href="0x000010af" shape=box ];
V_0x000010b1 [ label=<000010b1  ?? nop    dword ds:[rax + 0x00000000&lt;(func)__cxa_finalize@@GLIBC_2.2.5&gt;]<br align="left"/>> fontname=Courier href="0x000010b1" shape=box ];
}

subgraph cluster_0x000010c0 { label="function 0x000010c0 \\"register_tm_clones\\"" fillcolor="#f2f2f2" href="0x000010c0" style=filled
V_0x000010c0 [ label=<000010c0  ?? lea    rdi, [rip + 0x0000000000002f49&lt;12105,absolute=0x0000000000004010&gt;]<br align="left"/>000010c7  ?? lea    rsi, [rip + 0x0000000000002f42&lt;12098,absolute=0x0000000000004010&gt;]<br align="left"/>000010ce  ?? sub    rsi, rdi<br align="left"/>000010d1  ?? mov    rax, rsi<br align="left"/>000010d4  ?? shr    rsi, 0x3f<br align="left"/>000010d8  ?? sar    rax, 0x03<br align="left"/>000010dc  ?? add    rsi, rax<br align="left"/>000010df  ?? sar    rsi, 0x01<br align="left"/>000010e2  ?? je     0x00000000000010f8&lt;4344&gt;<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x000010c0" shape=box style=filled ];
V_0x000010e4 [ label=<000010e4  ?? mov    rax, qword ds:[rip + 0x0000000000002f05&lt;12037,absolute=0x0000000000003ff0&gt;]<br align="left"/>000010eb  ?? test   rax, rax<br align="left"/>000010ee  ?? je     0x00000000000010f8&lt;4344&gt;<br align="left"/>> fontname=Courier href="0x000010e4" shape=box ];
V_0x000010f8 [ label=<000010f8  ?? ret    <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x000010f8" shape=box style=filled ];
V_0x000010f0 [ label=<000010f0  ?? jmp    rax<br align="left"/>> fontname=Courier href="0x000010f0" shape=box ];
V_0x000010f2 [ label=<000010f2  ?? nop    word ds:[rax + rax + 0x00&lt;(func)__cxa_finalize@@GLIBC_2.2.5&gt;]<br align="left"/>> fontname=Courier href="0x000010f2" shape=box ];
}

subgraph cluster_0x00001100 { label="function 0x00001100 \\"__do_global_dtors_aux\\"" fillcolor="#f2f2f2" href="0x00001100" style=filled
V_0x00001100 [ label=<00001100  ?? nop    <br align="left"/>00001104  ?? cmp    byte ds:[rip + 0x0000000000002f05&lt;12037,absolute=0x0000000000004010&gt;], 0x00&lt;(func)__cxa_finalize@@GLIBC_2.2.5&gt;<br align="left"/>0000110b  ?? jne    0x0000000000001138&lt;4408&gt;<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x00001100" shape=box style=filled ];
V_0x0000110d [ label=<0000110d  ?? push   rbp<br align="left"/>0000110e  ?? cmp    qword ds:[rip + 0x0000000000002ee2&lt;12002,absolute=0x0000000000003ff8&gt;], 0x00&lt;(func)__cxa_finalize@@GLIBC_2.2.5&gt;<br align="left"/>00001116  ?? mov    rbp, rsp<br align="left"/>00001119  ?? je     0x0000000000001127&lt;4391&gt;<br align="left"/>> fontname=Courier href="0x0000110d" shape=box ];
V_0x00001138 [ label=<00001138  ?? ret    <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x00001138" shape=box style=filled ];
V_0x0000111b [ label=<0000111b  ?? mov    rdi, qword ds:[rip + 0x0000000000002ee6&lt;12006,absolute=0x0000000000004008&gt;]<br align="left"/>00001122  ?? call   0x0000000000001040&lt;4160&gt;<br align="left"/>> fontname=Courier href="0x0000111b" shape=box ];
V_0x00001127 [ label=<00001127  ?? call   0x0000000000001090&lt;4240,(func)deregister_tm_clones&gt;<br align="left"/>> fontname=Courier href="0x00001127" shape=box ];
V_0x0000112c [ label=<0000112c  ?? mov    byte ds:[rip + 0x0000000000002edd&lt;11997,absolute=0x0000000000004010&gt;], 0x01<br align="left"/>00001133  ?? pop    rbp<br align="left"/>00001134  ?? ret    <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x0000112c" shape=box style=filled ];
V_0x00001135 [ label=<00001135  ?? nop    dword ds:[rax]<br align="left"/>> fontname=Courier href="0x00001135" shape=box ];
}

subgraph cluster_0x00001140 { label="function 0x00001140 \\"frame_dummy\\"" fillcolor="#f2f2f2" href="0x00001140" style=filled
V_0x00001140 [ label=<00001140  ?? nop    <br align="left"/>00001144  ?? jmp    0x00000000000010c0&lt;4288,(func)register_tm_clones&gt;<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x00001140" shape=box style=filled ];
}

subgraph cluster_0x00001149 { label="function 0x00001149 \\"search_char\\"" fillcolor="#f2f2f2" href="0x00001149" style=filled
V_0x00001149 [ label=<00001149  ?? nop    <br align="left"/>0000114d  ?? push   rbp<br align="left"/>0000114e  ?? mov    rbp, rsp<br align="left"/>00001151  ?? mov    qword ds:[rbp + 0xe8&lt;-24&gt;], rdi<br align="left"/>00001155  ?? mov    eax, esi<br align="left"/>00001157  ?? mov    byte ds:[rbp + 0xe4&lt;-28&gt;], al<br align="left"/>0000115a  ?? mov    rax, qword ds:[rbp + 0xe8&lt;-24&gt;]<br align="left"/>0000115e  ?? mov    qword ds:[rbp + 0xf8&lt;-8&gt;], rax<br align="left"/>00001162  ?? jmp    0x0000000000001169&lt;4457&gt;<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x00001149" shape=box style=filled ];
V_0x00001174 [ label=<00001174  ?? mov    rax, qword ds:[rbp + 0xe8&lt;-24&gt;]<br align="left"/>00001178  ?? movzx  eax, byte ds:[rax]<br align="left"/>0000117b  ?? cmp    byte ds:[rbp + 0xe4&lt;-28&gt;], al<br align="left"/>0000117e  ?? jne    0x0000000000001164&lt;4452&gt;<br align="left"/>> fontname=Courier href="0x00001174" shape=box ];
V_0x00001180 [ label=<00001180  ?? mov    rax, qword ds:[rbp + 0xe8&lt;-24&gt;]<br align="left"/>00001184  ?? movzx  eax, byte ds:[rax]<br align="left"/>00001187  ?? test   al, al<br align="left"/>00001189  ?? jne    0x0000000000001191&lt;4497&gt;<br align="left"/>> fontname=Courier href="0x00001180" shape=box ];
V_0x0000118b [ label=<0000118b  ?? cmp    byte ds:[rbp + 0xe4&lt;-28&gt;], 0x00&lt;(func)__cxa_finalize@@GLIBC_2.2.5&gt;<br align="left"/>0000118f  ?? jne    0x000000000000119b&lt;4507&gt;<br align="left"/>> fontname=Courier href="0x0000118b" shape=box ];
V_0x00001191 [ label=<00001191  ?? mov    rax, qword ds:[rbp + 0xe8&lt;-24&gt;]<br align="left"/>00001195  ?? sub    rax, qword ds:[rbp + 0xf8&lt;-8&gt;]<br align="left"/>00001199  ?? jmp    0x00000000000011a0&lt;4512&gt;<br align="left"/>> fontname=Courier href="0x00001191" shape=box ];
V_0x0000119b [ label=<0000119b  ?? mov    eax, 0xffffffff&lt;-1&gt;<br align="left"/>> fontname=Courier href="0x0000119b" shape=box ];
V_0x000011a0 [ label=<000011a0  ?? pop    rbp<br align="left"/>000011a1  ?? ret    <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x000011a0" shape=box style=filled ];
V_0x00001164 [ label=<00001164  ?? add    qword ds:[rbp + 0xe8&lt;-24&gt;], 0x01<br align="left"/>> fontname=Courier href="0x00001164" shape=box ];
V_0x00001169 [ label=<00001169  ?? mov    rax, qword ds:[rbp + 0xe8&lt;-24&gt;]<br align="left"/>0000116d  ?? movzx  eax, byte ds:[rax]<br align="left"/>00001170  ?? test   al, al<br align="left"/>00001172  ?? je     0x0000000000001180&lt;4480&gt;<br align="left"/>> fontname=Courier href="0x00001169" shape=box ];
}

subgraph cluster_0x000011a2 { label="function 0x000011a2 \\"mutate\\"" fillcolor="#f2f2f2" href="0x000011a2" style=filled
V_0x000011a2 [ label=<000011a2  ?? nop    <br align="left"/>000011a6  ?? push   rbp<br align="left"/>000011a7  ?? mov    rbp, rsp<br align="left"/>000011aa  ?? sub    rsp, 0x30<br align="left"/>000011ae  ?? mov    qword ds:[rbp + 0xd8&lt;-40&gt;], rdi<br align="left"/>000011b2  ?? mov    dword ds:[rbp + 0xd4&lt;-44&gt;], esi<br align="left"/>000011b5  ?? mov    rax, qword ds:[rbp + 0xd8&lt;-40&gt;]<br align="left"/>000011b9  ?? mov    qword ds:[rbp + 0xf0&lt;-16&gt;], rax<br align="left"/>000011bd  ?? mov    rax, qword ds:[rbp + 0xd8&lt;-40&gt;]<br align="left"/>000011c1  ?? mov    esi, 0x00000000&lt;(func)__cxa_finalize@@GLIBC_2.2.5&gt;<br align="left"/>000011c6  ?? mov    rdi, rax<br align="left"/>000011c9  ?? call   0x0000000000001149&lt;4425,(func)search_char&gt;<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x000011a2" shape=box style=filled ];
V_0x000011ce [ label=<000011ce  ?? mov    dword ds:[rbp + 0xec&lt;-20&gt;], eax<br align="left"/>000011d1  ?? cmp    dword ds:[rbp + 0xec&lt;-20&gt;], 0x00&lt;(func)__cxa_finalize@@GLIBC_2.2.5&gt;<br align="left"/>000011d5  ?? jns    0x0000000000001216&lt;4630&gt;<br align="left"/>> fontname=Courier href="0x000011ce" shape=box ];
V_0x000011d7 [ label=<000011d7  ?? mov    eax, 0x00000001<br align="left"/>000011dc  ?? jmp    0x000000000000122d&lt;4653&gt;<br align="left"/>> fontname=Courier href="0x000011d7" shape=box ];
V_0x00001216 [ label=<00001216  ?? cmp    qword ds:[rbp + 0xf0&lt;-16&gt;], 0x00&lt;(func)__cxa_finalize@@GLIBC_2.2.5&gt;<br align="left"/>0000121b  ?? je     0x0000000000001228&lt;4648&gt;<br align="left"/>> fontname=Courier href="0x00001216" shape=box ];
V_0x0000121d [ label=<0000121d  ?? mov    rax, qword ds:[rbp + 0xd8&lt;-40&gt;]<br align="left"/>00001221  ?? movzx  eax, byte ds:[rax]<br align="left"/>00001224  ?? test   al, al<br align="left"/>00001226  ?? jne    0x00000000000011de&lt;4574&gt;<br align="left"/>> fontname=Courier href="0x0000121d" shape=box ];
V_0x00001228 [ label=<00001228  ?? mov    eax, 0x00000000&lt;(func)__cxa_finalize@@GLIBC_2.2.5&gt;<br align="left"/>> fontname=Courier href="0x00001228" shape=box ];
V_0x000011de [ label=<000011de  ?? mov    rax, qword ds:[rbp + 0xd8&lt;-40&gt;]<br align="left"/>000011e2  ?? movzx  eax, byte ds:[rax]<br align="left"/>000011e5  ?? movsx  ecx, al<br align="left"/>000011e8  ?? mov    eax, dword ds:[rbp + 0xd4&lt;-44&gt;]<br align="left"/>000011eb  ?? cdq    <br align="left"/>000011ec  ?? idiv   ecx<br align="left"/>000011ee  ?? mov    eax, edx<br align="left"/>000011f0  ?? cdq    <br align="left"/>000011f1  ?? idiv   dword ds:[rbp + 0xec&lt;-20&gt;]<br align="left"/>000011f4  ?? mov    eax, edx<br align="left"/>000011f6  ?? movsxd rdx, eax<br align="left"/>000011f9  ?? mov    rax, qword ds:[rbp + 0xf0&lt;-16&gt;]<br align="left"/>000011fd  ?? add    rax, rdx<br align="left"/>00001200  ?? mov    qword ds:[rbp + 0xf8&lt;-8&gt;], rax<br align="left"/>00001204  ?? mov    rax, qword ds:[rbp + 0xd8&lt;-40&gt;]<br align="left"/>00001208  ?? movzx  edx, byte ds:[rax]<br align="left"/>0000120b  ?? mov    rax, qword ds:[rbp + 0xf8&lt;-8&gt;]<br align="left"/>0000120f  ?? mov    byte ds:[rax], dl<br align="left"/>00001211  ?? add    qword ds:[rbp + 0xd8&lt;-40&gt;], 0x01<br align="left"/>> fontname=Courier href="0x000011de" shape=box ];
V_0x0000122d [ label=<0000122d  ?? leave  <br align="left"/>0000122e  ?? ret    <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x0000122d" shape=box style=filled ];
}

subgraph cluster_0x0000122f { label="function 0x0000122f \\"main\\"" fillcolor="#f2f2f2" href="0x0000122f" style=filled
V_0x0000122f [ label=<0000122f  ?? nop    <br align="left"/>00001233  ?? push   rbp<br align="left"/>00001234  ?? mov    rbp, rsp<br align="left"/>00001237  ?? sub    rsp, 0x20<br align="left"/>0000123b  ?? mov    dword ds:[rbp + 0xec&lt;-20&gt;], edi<br align="left"/>0000123e  ?? mov    qword ds:[rbp + 0xe0&lt;-32&gt;], rsi<br align="left"/>00001242  ?? mov    rax, qword ds:[rbp + 0xe0&lt;-32&gt;]<br align="left"/>00001246  ?? mov    rax, qword ds:[rax + 0x08]<br align="left"/>0000124a  ?? mov    qword ds:[rbp + 0xf8&lt;-8&gt;], rax<br align="left"/>0000124e  ?? mov    rax, qword ds:[rbp + 0xf8&lt;-8&gt;]<br align="left"/>00001252  ?? mov    esi, 0x0000003a<br align="left"/>00001257  ?? mov    rdi, rax<br align="left"/>0000125a  ?? call   0x0000000000001149&lt;4425,(func)search_char&gt;<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x0000122f" shape=box style=filled ];
V_0x0000125f [ label=<0000125f  ?? mov    dword ds:[rbp + 0xf0&lt;-16&gt;], eax<br align="left"/>00001262  ?? mov    eax, dword ds:[rbp + 0xf0&lt;-16&gt;]<br align="left"/>00001265  ?? mov    esi, eax<br align="left"/>00001267  ?? lea    rdi, [rip + 0x0000000000000d96&lt;3478,absolute=0x0000000000002004&gt;]<br align="left"/>0000126e  ?? mov    eax, 0x00000000&lt;(func)__cxa_finalize@@GLIBC_2.2.5&gt;<br align="left"/>00001273  ?? call   0x0000000000001050&lt;4176&gt;<br align="left"/>> fontname=Courier href="0x0000125f" shape=box ];
V_0x00001278 [ label=<00001278  ?? mov    rax, qword ds:[rbp + 0xf8&lt;-8&gt;]<br align="left"/>0000127c  ?? mov    rsi, rax<br align="left"/>0000127f  ?? lea    rdi, [rip + 0x0000000000000d8d&lt;3469,absolute=0x0000000000002013&gt;]<br align="left"/>00001286  ?? mov    eax, 0x00000000&lt;(func)__cxa_finalize@@GLIBC_2.2.5&gt;<br align="left"/>0000128b  ?? call   0x0000000000001050&lt;4176&gt;<br align="left"/>> fontname=Courier href="0x00001278" shape=box ];
V_0x00001290 [ label=<00001290  ?? mov    rax, qword ds:[rbp + 0xf8&lt;-8&gt;]<br align="left"/>00001294  ?? mov    esi, 0x00007ab7&lt;31415&gt;<br align="left"/>00001299  ?? mov    rdi, rax<br align="left"/>0000129c  ?? call   0x00000000000011a2&lt;4514,(func)mutate&gt;<br align="left"/>> fontname=Courier href="0x00001290" shape=box ];
V_0x000012a1 [ label=<000012a1  ?? mov    dword ds:[rbp + 0xf4&lt;-12&gt;], eax<br align="left"/>000012a4  ?? mov    rax, qword ds:[rbp + 0xf8&lt;-8&gt;]<br align="left"/>000012a8  ?? mov    rsi, rax<br align="left"/>000012ab  ?? lea    rdi, [rip + 0x0000000000000d6d&lt;3437,absolute=0x000000000000201f&gt;]<br align="left"/>000012b2  ?? mov    eax, 0x00000000&lt;(func)__cxa_finalize@@GLIBC_2.2.5&gt;<br align="left"/>000012b7  ?? call   0x0000000000001050&lt;4176&gt;<br align="left"/>> fontname=Courier href="0x000012a1" shape=box ];
V_0x000012bc [ label=<000012bc  ?? mov    eax, 0x00000000&lt;(func)__cxa_finalize@@GLIBC_2.2.5&gt;<br align="left"/>000012c1  ?? leave  <br align="left"/>000012c2  ?? ret    <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x000012bc" shape=box style=filled ];
}

subgraph cluster_0x000012d0 { label="function 0x000012d0 \\"__libc_csu_init\\"" fillcolor="#f2f2f2" href="0x000012d0" style=filled
V_0x000012d0 [ label=<000012d0  ?? nop    <br align="left"/>000012d4  ?? push   r15<br align="left"/>000012d6  ?? lea    r15, [rip + 0x0000000000002adb&lt;10971,absolute=0x0000000000003db8&gt;]<br align="left"/>000012dd  ?? push   r14<br align="left"/>000012df  ?? mov    r14, rdx<br align="left"/>000012e2  ?? push   r13<br align="left"/>000012e4  ?? mov    r13, rsi<br align="left"/>000012e7  ?? push   r12<br align="left"/>000012e9  ?? mov    r12d, edi<br align="left"/>000012ec  ?? push   rbp<br align="left"/>000012ed  ?? lea    rbp, [rip + 0x0000000000002acc&lt;10956,absolute=0x0000000000003dc0&gt;]<br align="left"/>000012f4  ?? push   rbx<br align="left"/>000012f5  ?? sub    rbp, r15<br align="left"/>000012f8  ?? sub    rsp, 0x08<br align="left"/>000012fc  ?? call   0x0000000000001000&lt;4096,(func)_init&gt;<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x000012d0" shape=box style=filled ];
V_0x00001301 [ label=<00001301  ?? sar    rbp, 0x03<br align="left"/>00001305  ?? je     0x0000000000001326&lt;4902&gt;<br align="left"/>> fontname=Courier href="0x00001301" shape=box ];
V_0x00001307 [ label=<00001307  ?? xor    ebx, ebx<br align="left"/>00001309  ?? nop    dword ds:[rax + 0x00000000&lt;(func)__cxa_finalize@@GLIBC_2.2.5&gt;]<br align="left"/>> fontname=Courier href="0x00001307" shape=box ];
V_0x00001326 [ label=<00001326  ?? add    rsp, 0x08<br align="left"/>0000132a  ?? pop    rbx<br align="left"/>0000132b  ?? pop    rbp<br align="left"/>0000132c  ?? pop    r12<br align="left"/>0000132e  ?? pop    r13<br align="left"/>00001330  ?? pop    r14<br align="left"/>00001332  ?? pop    r15<br align="left"/>00001334  ?? ret    <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x00001326" shape=box style=filled ];
V_0x0000131d [ label=<0000131d  ?? add    rbx, 0x01<br align="left"/>00001321  ?? cmp    rbp, rbx<br align="left"/>00001324  ?? jne    0x0000000000001310&lt;4880&gt;<br align="left"/>> fontname=Courier href="0x0000131d" shape=box ];
V_0x00001310 [ label=<00001310  ?? mov    rdx, r14<br align="left"/>00001313  ?? mov    rsi, r13<br align="left"/>00001316  ?? mov    edi, r12d<br align="left"/>00001319  ?? call   qword ds:[r15 + rbx*0x08]<br align="left"/>> fontname=Courier href="0x00001310" shape=box ];
}

subgraph cluster_0x00001340 { label="function 0x00001340 \\"__libc_csu_fini\\"" fillcolor="#f2f2f2" href="0x00001340" style=filled
V_0x00001340 [ label=<00001340  ?? nop    <br align="left"/>00001344  ?? ret    <br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x00001340" shape=box style=filled ];
}

subgraph cluster_0x00001348 { label="function 0x00001348 \\"_fini\\"" fillcolor="#f2f2f2" href="0x00001348" style=filled
V_0x00001348 [ label=<00001348  ?? nop    <br align="left"/>0000134c  ?? sub    rsp, 0x08<br align="left"/>00001350  ?? add    rsp, 0x08<br align="left"/>00001354  ?? ret    <br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x00001348" shape=box style=filled ];
}
indeterminate [ label="indeterminate" fillcolor="#ff9999" shape=box style=filled ];
V_0x00001050 -> indeterminate [ label="other"  ];
V_0x00001060 -> V_0x0000108e [ label="cret\\nassumed" style=dotted ];
V_0x00001000 -> V_0x00001016 [ label=""  ];
V_0x00001090 -> V_0x000010b8 [ label=""  ];
V_0x000010c0 -> V_0x000010f8 [ label=""  ];
V_0x00001100 -> V_0x00001138 [ label=""  ];
V_0x00001169 -> V_0x00001180 [ label=""  ];
V_0x00001180 -> V_0x00001191 [ label=""  ];
V_0x00001149 -> V_0x00001169 [ label=""  ];
V_0x00001174 -> V_0x00001180 [ label="" style=dotted ];
V_0x00001174 -> V_0x00001164 [ label=""  ];
V_0x00001191 -> V_0x000011a0 [ label=""  ];
V_0x00001180 -> V_0x0000118b [ label="" style=dotted ];
V_0x0000118b -> V_0x0000119b [ label=""  ];
V_0x0000118b -> V_0x00001191 [ label="" style=dotted ];
V_0x0000119b -> V_0x000011a0 [ label="" style=dotted ];
V_0x00001164 -> V_0x00001169 [ label="" style=dotted ];
V_0x00001169 -> V_0x00001174 [ label="" style=dotted ];
V_0x00001140 -> V_0x000010c0 [ label="other"  ];
V_0x0000110d -> V_0x00001127 [ label=""  ];
V_0x00001100 -> V_0x0000110d [ label="" style=dotted ];
V_0x00001014 -> V_0x00001016 [ label="cret\\nassumed" style=dotted ];
V_0x0000110d -> V_0x0000111b [ label="" style=dotted ];
V_0x0000111b -> V_0x00001127 [ label="cret" style=dotted ];
V_0x00001040 -> indeterminate [ label="other"  ];
V_0x000010e4 -> V_0x000010f8 [ label=""  ];
V_0x000010c0 -> V_0x000010e4 [ label="" style=dotted ];
V_0x000010e4 -> V_0x000010f0 [ label="" style=dotted ];
V_0x000010f0 -> indeterminate [ label="other"  ];
V_0x000010a3 -> V_0x000010b8 [ label=""  ];
V_0x00001090 -> V_0x000010a3 [ label="" style=dotted ];
V_0x000010a3 -> V_0x000010af [ label="" style=dotted ];
V_0x000010af -> indeterminate [ label="other"  ];
V_0x00001000 -> V_0x00001014 [ label="" style=dotted ];
V_0x0000111b -> V_0x00001040 [ label="call" color="#05ff00" ];
V_0x00001127 -> V_0x0000112c [ label="cret" style=dotted ];
V_0x00001127 -> V_0x00001090 [ label="call" color="#05ff00" ];
V_0x00001216 -> V_0x00001228 [ label=""  ];
V_0x000011a2 -> V_0x00001149 [ label="call" color="#05ff00" ];
V_0x000011a2 -> V_0x000011ce [ label="cret" style=dotted ];
V_0x00001228 -> V_0x0000122d [ label="" style=dotted ];
V_0x000011ce -> V_0x000011d7 [ label="" style=dotted ];
V_0x000011ce -> V_0x00001216 [ label=""  ];
V_0x0000121d -> V_0x00001228 [ label="" style=dotted ];
V_0x00001216 -> V_0x0000121d [ label="" style=dotted ];
V_0x000011de -> V_0x00001216 [ label="" style=dotted ];
V_0x0000121d -> V_0x000011de [ label=""  ];
V_0x000011d7 -> V_0x0000122d [ label=""  ];
V_0x0000122f -> V_0x0000125f [ label="cret" style=dotted ];
V_0x0000122f -> V_0x00001149 [ label="call" color="#05ff00" ];
V_0x0000125f -> V_0x00001278 [ label="cret" style=dotted ];
V_0x0000125f -> V_0x00001050 [ label="call" color="#05ff00" ];
V_0x00001278 -> V_0x00001290 [ label="cret" style=dotted ];
V_0x00001278 -> V_0x00001050 [ label="call" color="#05ff00" ];
V_0x00001290 -> V_0x000012a1 [ label="cret" style=dotted ];
V_0x00001290 -> V_0x000011a2 [ label="call" color="#05ff00" ];
V_0x000012a1 -> V_0x000012bc [ label="cret" style=dotted ];
V_0x000012a1 -> V_0x00001050 [ label="call" color="#05ff00" ];
V_0x00001301 -> V_0x00001326 [ label=""  ];
V_0x000012d0 -> V_0x00001000 [ label="call" color="#05ff00" ];
V_0x000012d0 -> V_0x00001301 [ label="cret" style=dotted ];
V_0x00001301 -> V_0x00001307 [ label="" style=dotted ];
V_0x0000131d -> V_0x00001310 [ label=""  ];
V_0x00001060 -> indeterminate [ label="call" color="#05ff00" ];
V_0x00001014 -> indeterminate [ label="call" color="#05ff00" ];
V_0x00001307 -> V_0x00001310 [ label="" style=dotted ];
V_0x0000131d -> V_0x00001326 [ label="" style=dotted ];
V_0x00001310 -> indeterminate [ label="call" color="#05ff00" ];
V_0x00001310 -> V_0x0000131d [ label="cret\\nassumed" style=dotted ];
V_0x000010f2 -> V_0x000010f8 [ label="" style=dotted ];
V_0x00001135 -> V_0x00001138 [ label="" style=dotted ];
V_0x000010b1 -> V_0x000010b8 [ label="" style=dotted ];
}
"""