#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include <stdio.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <argp.h>
#include <cassert>
#include <queue>
#include "Symtab.h"
#include "Function.h"

using namespace std;
using namespace Dyninst;
using namespace SymtabAPI;

#define APP_NAME "symt_addr2line"
#define APP_DESC "addr2line clone based off of SymtabAPI"
#define APP_BUG_ADDRESS "legendre1@llnl.gov"
#define APP_VERSION "0.1"

class CmdLine {
private:
   std::string executable;
   bool truncate_dirs;
   bool show_functions;
   bool do_demangle;
   bool has_addrs;
   std::queue<unsigned long> addrs;
   Symtab *symbols;
   static CmdLine *me;

   bool isFile(string file) const;
   error_t parseArg(int key, char *arg, void *vstate);
   static error_t parseWrapper(int key, char *arg, struct argp_state *state);
public:
   CmdLine(int argc, char *argv[]);
   std::string getExecutable() const;
   bool truncateDirs() const;
   bool showFunctions() const;
   bool demangle() const;
   bool hasAddresses() const;
   bool getNextAddress(unsigned long &addr);
   Symtab *getSymtab() const;
};

class Addr2Line {
   CmdLine *cmdline;
   Symtab *symbols;
private:
   bool getNextAddress(unsigned long &addr);
   bool printFunction(unsigned long addr);
   bool printLine(unsigned long addr);
public:
   Addr2Line(CmdLine *cl);
   void run();
};

int main(int argc, char *argv[])
{
   CmdLine cmdline(argc, argv);
   Addr2Line al(&cmdline);

   al.run();
   
   return 0;
}

bool Addr2Line::getNextAddress(unsigned long &addr) 
{
   if (cmdline->hasAddresses()) {
      //Read from command line
      return cmdline->getNextAddress(addr);
   }
   
   //Read from stdin
   char *lineptr = NULL;
   size_t linesize = 0;
   ssize_t result = getline(&lineptr, &linesize, stdin);
   if (result == -1)
      return false;
   char *c = lineptr;
   while (*c == ' ' && *c == '\t') c++;
   if (c[0] == '0' && c[1] == 'x') c += 2;
   sscanf(c, "%lx", &addr);
   free(lineptr);
   return true;
}

bool Addr2Line::printFunction(unsigned long addr) 
{
   Function *func = NULL;
   bool result = symbols->getContainingFunction(addr, func);
   if (!result || !func) {
      printf("??\n");
      return false;
   }
   
   const vector<string> &names = cmdline->demangle() ? func->getAllTypedNames() : func->getAllMangledNames();
   if (names.empty()) {
      printf("??\n");
      return false;
   }
   printf("%s\n", names[0].c_str());
   return true;
}

bool Addr2Line::printLine(unsigned long addr) 
{
   std::vector<Statement *> lines;
   Statement *line;
   bool result;
   
   result = symbols->getSourceLines(lines, addr);
   if (!result || lines.empty())
      goto err;
   
   line = lines[0];
   if (!line) 
      goto err;
   
   printf("%s:%u\n", line->getFile().c_str(), line->getLine());
   return true;
  err:
   printf("??:0\n");
   return false;
}

void Addr2Line::run() 
{
   unsigned long addr;
   while (getNextAddress(addr)) {
      if (cmdline->showFunctions())
         printFunction(addr);
      printLine(addr);
   }
}

Addr2Line::Addr2Line(CmdLine *cl) :
   cmdline(cl)
{
   symbols = cmdline->getSymtab();
}

#define TARGET 'b'
#define EXE 'e'
#define INLINES 'i'
#define SECTION 'j'
#define BASENAMES 's'
#define FUNCTIONS 'f'
#define DEMANGLE 'C'
#define VERSION 'v'

struct argp_option aoptions[] = {
   { "target", TARGET, "bfdname", 0, "Set the binary format", 0},
   { "exe", EXE, "executable", 0, "Set the input file name (default is a.out)", 0},
   { "inlines", INLINES, NULL, 0, "Unwind inlined functions", 0},
   { "section", SECTION, "name", 0, "Read section-relative offsets instead of addresses", 0},
   { "basenames", BASENAMES, NULL, 0, "Strip directory names", 0},
   { "functions", FUNCTIONS, NULL, 0, "Show function names", 0},
   { "demangle", DEMANGLE, "style", OPTION_ARG_OPTIONAL, "Demangle function names", 0},
   { "version", VERSION, NULL, 0, "Display the program's version", 0},
   {0}
};


CmdLine *CmdLine::me = NULL;

bool CmdLine::isFile(string file) const {
   struct stat buf;
   int result = stat(file.c_str(), &buf);
   if (result == -1) {
      return false;
   }
   if (S_ISDIR(buf.st_mode)) {
      return false;
   }
   return true;
}

error_t CmdLine::parseArg(int key, char *arg, void *vstate) {
   struct argp_state *state = (struct argp_state *) vstate;
   switch (key) {
      case TARGET:
      case INLINES:
      case SECTION:
         /* Ignored in this version */
         break;
      case EXE:
         executable = arg;
         break;
      case BASENAMES:
         truncate_dirs = true;
         break;
      case FUNCTIONS:
         show_functions = true;
         break;
      case DEMANGLE:
         do_demangle = true;
         break;
      case VERSION:
         printf("%s v%s\n%s\n", APP_NAME, APP_VERSION, APP_DESC);
         exit(0);
         break;
      case ARGP_KEY_ARG: {
         char *hex = arg;
         if (arg[0] == '0' && arg[1] == 'x') {
            hex = arg+2;
         }
         unsigned long hex_num = 0;
         sscanf(hex, "%lx", &hex_num);
         addrs.push(hex_num);
         has_addrs = true;
      }
      case ARGP_KEY_END: {
         if (!isFile(executable)) {
            argp_error(state, "%s: '%s': No such file\n", APP_NAME, executable.c_str());
         }
         bool result = Symtab::openFile(symbols, executable);
         if (!result || !symbols) {
            argp_error(state, "%s: Could parse file %s\n", APP_NAME, executable.c_str());
         }
         symbols->setTruncateLinePaths(truncate_dirs);
         break;
      }
   }
   return 0;
}

error_t CmdLine::parseWrapper(int key, char *arg, struct argp_state *state) {
   return me->parseArg(key, arg, state);
}

CmdLine::CmdLine(int argc, char *argv[]) :
   executable("a.out"),
   truncate_dirs(false),
   show_functions(false),
   do_demangle(false),
   has_addrs(false)
{
   assert(!me);
   me = this;
   argp_program_version = APP_VERSION;
   argp_program_bug_address = APP_BUG_ADDRESS;
   
   struct argp arg_parser;
   bzero(&arg_parser, sizeof(struct argp));
   arg_parser.options = aoptions;
   arg_parser.parser = parseWrapper;
   
   error_t result = argp_parse(&arg_parser, argc, argv, 0, NULL, NULL);
   assert(result == 0);
}

std::string CmdLine::getExecutable() const 
{ 
   return executable;
}

bool CmdLine::truncateDirs() const 
{ 
   return truncate_dirs;
}

bool CmdLine::showFunctions() const 
{ 
   return show_functions;
}

bool CmdLine::demangle() const 
{
   return do_demangle;
}

bool CmdLine::hasAddresses() const
{ 
   return has_addrs;
}

bool CmdLine::getNextAddress(unsigned long &addr) 
{
   if (addrs.empty())
      return false;
   addr = addrs.front();
   addrs.pop();
   return true;
}

Symtab *CmdLine::getSymtab() const
{
   return symbols;
}
