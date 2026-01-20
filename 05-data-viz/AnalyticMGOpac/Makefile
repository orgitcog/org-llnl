
opacTest: *.hh *.cc
	clang++ -std=c++20 -Wall -Wextra -Werror -O3 MultiGroupIntegrator.cc AnalyticEdgeOpacity.cc opacTest.cc -o opacTest

opacTestSan: *.hh *.cc
	clang++ -std=gnu++20 -Wall -Wextra -Werror -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer  MultiGroupIntegrator.cc AnalyticEdgeOpacity.cc opacTest.cc -o opacTestSan

format:
	clang-format -i *.cc *.hh

clean:
	rm -rf opacTest opacTestSan *.dSYM
