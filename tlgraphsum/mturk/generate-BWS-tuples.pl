#!/usr/bin/perl

################################################################################################################################
#  Authors: Svetlana Kiritchenko, Peter Turney
#  Information and Communications Technologies / Technologies de l'information et des communications
#  National Research Council Canada /Conseil national de recherches Canada
#  
#  Description: generates a set of tuples for Best-Worst Scaling annotation
#
#  Usage: generate-BWS-tuples.pl <file-items>
#    <file-items> is a file that contains a list of items to be annotated (one item per line)
#
#  Output: a list of item tuples (one tuple per line; items in a tuple are separated by tab).
#     The output is written into file <file-items>.tuples
#  
#  Version: 1.2
#  Last modified: Sep. 29, 2016
#
#################################################################################################################################

use warnings;
use strict;
use utf8;
binmode (STDIN,  ":utf8");
binmode (STDOUT, ":utf8");

use List::Util qw( shuffle );

#################################################################################################################################
# PARAMETERS
#################################################################################################################################

# number of items per tuple (typically, 4 or 5)
my $items_per_tuple = 4; 

# Best-Worst Scaling factor (typically 1.5 or 2):
#   multiply the number of items in $file_items by this factor
#   in order to determine the number of tuples to generate
my $factor = 2;  

# number of iterations (typically 100 or 1000)
my $num_iter = 100;


#################################################################################################################################

# random number seed (to make results reproducible)
my $rand_seed = 1234;
srand($rand_seed);


# read the input file with the list of items (terms)
print STDERR "Reading ... \n";

my %unique_items = ();
while (my $line = <>) {
	$line =~ s/[\r\n]+$//;    # remove end-of-line characters

	# check for duplicate items
	if(defined $unique_items{$line}) {
		print STDERR "WARNING: duplicate item ($line); will be included only once.\n";
		next;
	}
	
	$unique_items{$line} = 1;
}

my @items = sort keys %unique_items;
my $num_items = scalar(@items);
my $num_unique_pairs = ($num_items * ($num_items - 1)) / 2;
print STDERR "Read $num_items unique items.\n\n";

# check if the number of unique items is not less than the number of items requested per tuple
if($num_items < $items_per_tuple) {
	print STDERR "ERROR: The number of unique items is less than the number of items requested per tuple\n";
	exit();
}


# generate tuples
my $num_tuples = int(0.5 + ($factor * $num_items));
print STDERR "Generating ".$num_tuples." ".$items_per_tuple."-tuples ...\n";

# try $num_iter different randomizations
print STDERR "Running $num_iter iterations ...\n";

my $best_score;
my @best_tuples = ();

for (my $iter = 1; $iter <= $num_iter; $iter++) {
	print STDERR "iteration $iter\n";

	# generate $num_tuples tuples by randomly sampling without replacement
	my @tuples = ();
	my @ranlist = shuffle(@items);     # make a random list of items
	my %freq_pair = ();
	
	my $j = 0;   # index of the current item in the random list
	for (my $i = 0; $i < $num_tuples; $i++) {
	
		my @tuple = ();   # new tuple
		
		# check if we have enough remained items in the random list to form a new tuple
		if(($j + $items_per_tuple) <= @ranlist) {
		
			# form a new tuple with $items_per_tuple items in the random list starting at index $j
			push(@tuple, @ranlist[$j..$j+$items_per_tuple-1]);
			$j += $items_per_tuple;
			
		} else {   
			# get the rest of the list
			my %items = ();
			my $need_more = $items_per_tuple - scalar(@ranlist) + $j;  # the number of items that we will need to get from a new random list
			for(; $j < @ranlist; $j++) {
				push(@tuple, $ranlist[$j]);
				$items{$ranlist[$j]} = 1;
			}
			
			# generate a new random list of items
			@ranlist = shuffle(@items);
			for($j = 0; $j < $need_more; $j++) {
			
				# if a duplicate item, move it to the end of the list
				while(defined $items{$ranlist[$j]}) {
					my $h = splice(@ranlist, $j, 1);
					push(@ranlist, $h);
				}
				push(@tuple, $ranlist[$j]);
			}			
		}
		
		my $tuple_string = join("\t", @tuple);
		push(@tuples, $tuple_string);

		# add frequencies of pairs of items
		for(my $k1 = 0; $k1 < @tuple; $k1++) {
			for(my $k2 = $k1+1; $k2 < @tuple; $k2++) {
				if($tuple[$k1] lt $tuple[$k2]) {
					$freq_pair{$tuple[$k1]."::".$tuple[$k2]}++;
				} else {
					$freq_pair{$tuple[$k2]."::".$tuple[$k1]}++;
				}
			}
		}
	}	

	# calculate the two-way balance of the set of tuples
	my @freq_pair_values = values %freq_pair;
	my $stddev_pairs = stdev(\@freq_pair_values, $num_unique_pairs);

	# calculate the score for the set and keep the best score and the best set
	my $score = $stddev_pairs;
  
	if (($iter == 1) || ($score < $best_score)) {
		$best_score = $score;
		@best_tuples = @tuples;
	}  
}

# output the best set of tuples to $file_output

foreach my $t (@best_tuples) {
  print "$t\n";
}

print STDERR "Finished.\n";



# calculate the standard deviation of a set of values
sub stdev{
	my($x, $n_total) = @_;
	my($n, $m, $sum, $i, $std);
	
	if($n_total == 1) {
		return 0;
	}
	
	$n = scalar(@{$x});

	$m = mean($x, $n_total);
	
	$sum = 0;
	for($i = 0; $i < $n; $i++) {
		$sum += ($m - $x->[$i]) ** 2;
	}
	
	$sum += $m * $m * ($n_total - $n);
	
	$std = sqrt($sum / ($n_total - 1));
	return $std;
}

# calculate the mean of a set of values
sub mean {
	my($x, $n_total) = @_;
	my($n, $i, $sum);
	
	$n = scalar(@{$x});
	
	for($i = 0; $i < $n; $i++) {
		$sum += $x->[$i];
	}
	
	return $sum/$n_total;
}

