#!/bin/bash

SCRIPT=${1:-hulk.py}
WORKSPACE=/tmp/$SCRIPT.$(id -u)
FAILURES=0

error() {
    echo "$@"
    [ -s $WORKSPACE/test ] && (echo ; cat $WORKSPACE/test; echo; rm $WORKSPACE/test)
    FAILURES=$((FAILURES + 1))
}

cleanup() {
    STATUS=${1:-$FAILURES}
    rm -fr $WORKSPACE
    exit $STATUS
}

mkdir $WORKSPACE

trap "cleanup" EXIT
trap "cleanup 1" INT TERM

echo "Testing $SCRIPT ..."

printf "   %-40s ... " "Unit Tests"
./hulk_test.py -v &> $WORKSPACE/test
TOTAL=$(grep 'Ran.*tests' $WORKSPACE/test | awk '{print $2}')
PASSED=$(grep -c '... ok' $WORKSPACE/test)
UNITS=$(echo "scale=2; ($PASSED / $TOTAL) * 2.0" | bc)
echo "$UNITS / 2.00"

printf "   %-40s ... " "Usage"
if ! ./hulk.py -h 2>&1 | grep -q -i usage > /dev/null; then
    error "Failure"
else
    echo "Success"
fi

printf "   %-40s ... " "Hulk LENGTH 1"
./$SCRIPT -s hashes.txt -l 1 > $WORKSPACE/test
if [ $(wc -l < $WORKSPACE/test) -ne 36 ]; then
    error "Failure"
else
    echo "Success"
fi

printf "   %-40s ... " "Hulk LENGTH 1 (ALPHABET: abc)"
./$SCRIPT -s hashes.txt -l 1 -a abc > $WORKSPACE/test
if [ $(wc -l < $WORKSPACE/test) -ne 3 ]; then
    error "Failure"
else
    echo "Success"
fi

printf "   %-40s ... " "Hulk LENGTH 2"
./$SCRIPT -s hashes.txt -l 2 > $WORKSPACE/test
if [ $(wc -l < $WORKSPACE/test) -ne 92 ]; then
    error "Failure"
else
    echo "Success"
fi

printf "   %-40s ... " "Hulk LENGTH 2 (ALPHABET: uty)"
./$SCRIPT -s hashes.txt -l 2 -a uty > $WORKSPACE/test
if [ $(wc -l < $WORKSPACE/test) -ne 6 ]; then
    error "Failure"
else
    echo "Success"
fi

printf "   %-40s ... " "Hulk LENGTH 3"
./$SCRIPT -s hashes.txt -l 3 > $WORKSPACE/test
if [ $(wc -l < $WORKSPACE/test) -ne 572 ]; then
    error "Failure"
else
    echo "Success"
fi

printf "   %-40s ... " "Hulk LENGTH 3 (ALPHABET: abc)"
./$SCRIPT -s hashes.txt -l 3 -a abc > $WORKSPACE/test
if [ $(wc -l < $WORKSPACE/test) -ne 7 ]; then
    error "Failure"
else
    echo "Success"
fi

printf "   %-40s ... " "Hulk LENGTH 4"
./$SCRIPT -s hashes.txt -l 4 > $WORKSPACE/test
if [ $(wc -l < $WORKSPACE/test) -ne 2654 ]; then
    error "Failure"
else
    echo "Success"
fi

printf "   %-40s ... " "Hulk LENGTH 4 (ALPHABET: uty)"
./$SCRIPT -s hashes.txt -l 4 -a uty > $WORKSPACE/test
if [ $(wc -l < $WORKSPACE/test) -ne 6 ]; then
    error "Failure"
else
    echo "Success"
fi

printf "   %-40s ... " "Hulk LENGTH 2 (CORES: 2)"
./$SCRIPT -s hashes.txt -l 2 -c 2 > $WORKSPACE/test
if [ $(wc -l < $WORKSPACE/test) -ne 92 ]; then
    error "Failure"
else
    echo "Success"
fi

printf "   %-40s ... " "Hulk LENGTH 3 (CORES: 2)"
./$SCRIPT -s hashes.txt -l 3 -c 2 > $WORKSPACE/test
if [ $(wc -l < $WORKSPACE/test) -ne 572 ]; then
    error "Failure"
else
    echo "Success"
fi

printf "   %-40s ... " "Hulk LENGTH 4 (CORES: 2)"
./$SCRIPT -s hashes.txt -l 4 -c 2 > $WORKSPACE/test
if [ $(wc -l < $WORKSPACE/test) -ne 2654 ]; then
    error "Failure"
else
    echo "Success"
fi

printf "   %-40s ... " "Hulk LENGTH 1 (PREFIX: a)"
./$SCRIPT -s hashes.txt -l 1 -p a > $WORKSPACE/test
if [ $(wc -l < $WORKSPACE/test) -ne 5 ]; then
    error "Failure"
else
    echo "Success"
fi

printf "   %-40s ... " "Hulk LENGTH 1 (PREFIX: 1, CORES: 2)"
./$SCRIPT -s hashes.txt -l 1 -p a -c 2 > $WORKSPACE/test
if [ $(wc -l < $WORKSPACE/test) -ne 5 ]; then
    error "Failure"
else
    echo "Success"
fi

printf "   %-40s ... " "Hulk LENGTH 2 (PREFIX: a)"
./$SCRIPT -s hashes.txt -l 2 -p a > $WORKSPACE/test
if [ $(wc -l < $WORKSPACE/test) -ne 53 ]; then
    error "Failure"
else
    echo "Success"
fi

printf "   %-40s ... " "Hulk LENGTH 2 (PREFIX: a, CORES: 2)"
./$SCRIPT -s hashes.txt -l 2 -p a -c 2 > $WORKSPACE/test
if [ $(wc -l < $WORKSPACE/test) -ne 53 ]; then
    error "Failure"
else
    echo "Success"
fi

printf "   %-40s ... " "Hulk LENGTH 3 (PREFIX: a)"
./$SCRIPT -s hashes.txt -l 3 -p a > $WORKSPACE/test
if [ $(wc -l < $WORKSPACE/test) -ne 185 ]; then
    error "Failure"
else
    echo "Success"
fi

printf "   %-40s ... " "Hulk LENGTH 3 (PREFIX: a, CORES: 2)"
./$SCRIPT -s hashes.txt -l 3 -p a -c 2 > $WORKSPACE/test
if [ $(wc -l < $WORKSPACE/test) -ne 185 ]; then
    error "Failure"
else
    echo "Success"
fi

TESTS=$(($(grep -c Success $0) - 1))

echo
echo "   Score $(echo "scale=2; $UNITS + ($TESTS - $FAILURES) / $TESTS.0 * 8.0" | bc)"
echo
