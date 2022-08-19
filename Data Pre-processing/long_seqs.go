package main

import (
	"encoding/csv"
	"io"
	"log"
	"strconv"
)

/*
This function handles sequences that will have portions dropped from them, salvaging sequences that can be padded.
Example : 5378 length sequence will have the 378 divided across the divisions
*/
func long_drop(r *csv.Reader, w io.Writer) int {
	count := 0
	writer := csv.NewWriter(w)
	writer.Comma = '\t'
	defer writer.Flush()

	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}

		start, _ := strconv.Atoi(record[1])
		end, _ := strconv.Atoi(record[2])
		length := end - start

		name := record[0]
		num_seqs := int(length / 1000) // Number of divisions in the sample
		remainder := length % 1000     // Remainder of sequence
		mid_seq := int(num_seqs / 2)   // Finding the middle of the divisions

		if length < 1000 { // Skip all sequences that are too small
			continue
		} else if length%1000 == 0 { // If the sequences is an even multiple of 1000, handle separately
			count += thousand_length(w, name, start, num_seqs)
		} else if length < 1600 {
			new_start := start
			new_end := new_start + 1000
			count += write(w, name, new_start, new_end, new_end-new_start)
			// Accounting for issues with shorter sequences
			// Breaking these short sequences into two 800 length chunks to preserve as much data as possible
		} else if length >= 1600 && length < 2000 {
			count += split_seq(w, name, length, start)
		} else {
			// This section does the sequence dropping
			if remainder < 800 {
				// This section drops sequence between 1000 length chunks.
				count += drop_seq(w, name, start, mid_seq, num_seqs, remainder)
			} else {
				count += padding_seq(w, name, start, mid_seq, num_seqs, remainder)
			}
		}
	}
	writer.Flush()
	return count
}

/*
This function dynamically distributes leftover sequences as overlappted portions throughout the sequence chunks.
Example : 3159 length seuqnce takes the 159 and spreads that value across divisions, so the beginning/end of sequences share nucleotides.
*/
func long_overlap(r *csv.Reader, w io.Writer) int {
	count := 0
	writer := csv.NewWriter(w)
	writer.Comma = '\t'
	defer writer.Flush()

	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}

		start, _ := strconv.Atoi(record[1])
		end, _ := strconv.Atoi(record[2])
		length := end - start

		name := record[0]
		num_seqs := int(length / 1000)    // Number of divisions in the sample
		remainder := length % 1000        // Remainder of sequence
		mid_seq := int(num_seqs / 2)      // Finding the middle of the divisions
		total_overlap := 1000 - remainder // 1000 - remainder to get the amount that needs to be spread across divisions
		var overlap int
		if num_seqs > 0 {
			overlap = int(total_overlap / num_seqs) // The amount of overlap between each sequence
		}

		if length < 1000 { // Skip all sequences that are too small
			continue
		} else if length%1000 == 0 { // If a multiple of 1000, break into equal 1000 chunks
			count += thousand_length(w, name, start, num_seqs)
		} else if length < 1600 { // Checking for sequences too short to salvage with other methods and only saving the first 1000 bases
			new_start := start
			new_end := new_start + 1000
			count += write(w, name, new_start, new_end, new_end-new_start)
		} else if length >= 1600 && length < 2000 { // Handling shorter sequences by breaking them into two sequences to be padded
			count += split_seq(w, name, length, start)
		} else if remainder >= 800 { // Checking for remainder long enough to pad
			count += padding_seq(w, name, start, mid_seq, num_seqs, remainder)
		} else if overlap > 100 { // If overlap is more than 100, drop the remainding sequence instead; 100 because it's on both sides of middle sequences
			count += drop_seq(w, name, start, mid_seq, num_seqs, remainder)
		} else { // If it makes it here, then the remainding sequence is short enough to overlap between sequences
			count += overlap_seq(w, name, start, mid_seq, num_seqs, total_overlap, overlap)
		}
	}
	writer.Flush()
	return count
}

/*
This function handles all the writing to the output file.
*/
func write(w io.Writer, name string, start int, end int, length int) int {
	writer := csv.NewWriter(w)
	writer.Comma = '\t'
	defer writer.Flush()
	// rec := []string{name, strconv.Itoa(start), strconv.Itoa(end), strconv.Itoa(length)} // FOR VERIFYING LENGTHS IN TESTING
	rec := []string{name, strconv.Itoa(start), strconv.Itoa(end)}
	if err := writer.Write(rec); err != nil {
		log.Fatalln("error writing record to csv:", err)
	}
	return 1
}

/*
This function takes sequences between 1600 and 2000 and splits them in half for padding.
*/
func split_seq(w io.Writer, name string, length int, start int) int {
	count := 0
	half := int(length / 2)
	new_start := start
	new_end := new_start + half
	count += write(w, name, new_start, new_end, new_end-new_start)
	new_start = new_start + half
	new_end = new_start + half + (length - (half * 2))
	count += write(w, name, new_start, new_end, new_end-new_start)
	return count
}

/*
This function handles all sequences that are multiples of 1000.
*/
func thousand_length(w io.Writer, name string, start int, num_seqs int) int {
	count := 0
	for i := 0; i < num_seqs; i++ {
		new_start := start + 1000*i
		new_end := new_start + 1000
		count += write(w, name, new_start, new_end, new_end-new_start)
	}
	return count
}

/*
This function handles all the sequences that have remainders of 800+, so they can be padded.
*/
func padding_seq(w io.Writer, name string, start int, mid_seq int, num_seqs int, remainder int) int {
	count := 0
	for i := 0; i <= num_seqs; i++ {
		new_start := 0
		new_end := 0
		if i == 0 {
			new_start = start
			new_end = new_start + 1000
		} else if i < mid_seq {
			new_start = 1000*i + start
			new_end = new_start + 1000
		} else if i == mid_seq { // Padding middle sequence, since motifs are generally at beginning/end
			new_start = 1000*i + start
			new_end = new_start + remainder
		} else {
			new_start = 1000*(i-1) + start + remainder
			new_end = new_start + 1000
		}
		count += write(w, name, new_start, new_end, new_end-new_start)
	}
	return count
}

/*
This function handles the sequences that need to have the remainder dropped across divisions.
*/
func drop_seq(w io.Writer, name string, start int, mid_seq int, num_seqs int, remainder int) int {
	count := 0
	remainder_div := int(remainder / (num_seqs - 1)) // -1 because we're losing the last chunk of the sequence
	for i := 0; i < num_seqs; i++ {
		new_start := 0
		new_end := 0
		if i == 0 {
			new_start = start
			new_end = new_start + 1000
		} else if i == mid_seq {
			start = start + (remainder - (remainder_div * (num_seqs - 1))) // Accounting for the missed bases by adding them to the middle drop
			new_start = 1000*i + remainder_div*i + start                   // *i to account for the amount of iterations
			new_end = new_start + 1000
		} else {
			new_start = 1000*i + remainder_div*i + start
			new_end = new_start + 1000
		}
		count += write(w, name, new_start, new_end, new_end-new_start)
	}
	return count
}

/*
This function handles sequences that have less than 100 bases being overlapped between any two sequences.
*/
func overlap_seq(w io.Writer, name string, start int, mid_seq int, num_seqs int, total_overlap int, overlap int) int {
	count := 0
	for i := 0; i <= num_seqs; i++ {
		new_start := 0
		new_end := 0
		if i == 0 {
			new_start = start
			new_end = new_start + 1000
		} else if i == mid_seq {
			start = start - (total_overlap - (overlap * (num_seqs))) // Accounting for the missed bases by adding them to the middle drop
			new_start = 1000*i - overlap*i + start
			new_end = new_start + 1000
		} else {
			new_start = 1000*i - overlap*i + start
			new_end = new_start + 1000
		}
		count += write(w, name, new_start, new_end, new_end-new_start)
	}
	return count
}
