package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
)

/*
Used to verify the lengths of the sequences are correct
*/
func check_length(seq string) {
	if len(seq) > 1000 || len(seq) < 1000 {
		fmt.Println("INCORRECT LENGTH: ", len(seq))
	}
}

/*
For saving the parsed validation data
*/
func validation_write(w io.Writer, name string, seq string, label string) int {
	writer := csv.NewWriter(w)
	writer.Comma = '\t'
	defer writer.Flush()
	rec := []string{name, seq, label}
	if err := writer.Write(rec); err != nil {
		log.Fatalln("error writing record to csv:", err)
	}
	return 1
}

/*
Parses sequences in TSV files into multiple 1000 length sequences
*/
func drop_tsv(in string, out string, pad bool, overlap bool) int {
	// Tracking number of CRMs
	count := 0

	// Creates file to write to
	w, err := os.Create(out)
	if err != nil {
		log.Fatal(err)
	}
	// Opens file
	file, err := os.Open(in)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	// Reads file
	scanner := bufio.NewScanner(file)

	// Parsing through file, line by line, breaking sequences into 1000 base chunks and padding 800+ length sequences along the way
	for scanner.Scan() {
		record := scanner.Text()
		indexes := strings.Split(record, "\t")
		seq := indexes[1]
		if len(seq) >= 1000 { // Loops through the number of divisions and drops unusable sequence between them
			// divs is the number of divisions in the sequence and remainder is the amount leftover
			divs := int(len(seq) / 1000)
			remainder := len(seq) % 1000
			if divs > 1 && remainder < 800 { // > 1, because this is where sequences are being dropped
				drops := int(remainder / (divs - 1))
				for i := 0; i < divs; i++ {
					if i >= int(divs/2) { // Adding missed bases to every division after the middle, to account for the chance in incrementing
						leftover := remainder - (drops * (divs - 1))
						// Adding basses missed in the casting process of finding the number of divisions
						start := (i * 1000) + leftover + drops
						stop := start + 1000
						new_seq := seq[start:stop]
						count += validation_write(w, indexes[0], new_seq, indexes[2])
						check_length(new_seq)
					} else { // Dropping sequence between 1000 base chunks
						start := (i * 1000) + drops
						stop := start + 1000
						new_seq := seq[start:stop]
						count += validation_write(w, indexes[0], new_seq, indexes[2])
						check_length(new_seq)
					}
				}
			} else if divs > 1 && remainder >= 800 {
				for i := 0; i < divs; i++ {
					start := i * 1000
					stop := start + 1000
					new_seq := seq[start:stop]
					count += validation_write(w, indexes[0], new_seq, indexes[2])
					check_length(new_seq)
					if i == divs-1 {
						new_seq = seq[stop:]
						new_seq = pad_seq(new_seq)
						count += validation_write(w, indexes[0], new_seq, indexes[2])
						check_length(new_seq)
					}
				}
			} else if remainder >= 800 { // If sequences >= 1000, with remainder >= 800, and less than 2000 bases long, save first 1000 and pad/save second part
				new_seq := seq[0:1000]
				count += validation_write(w, indexes[0], new_seq, indexes[2])
				check_length(new_seq)
				new_seq = seq[1000:]
				new_seq = pad_seq(new_seq)
				count += validation_write(w, indexes[0], new_seq, indexes[2])
				check_length(new_seq)
			} else { // If sequences >= 1000 and remainder is < 800, just save first 1000 bases
				new_seq := seq[0:1000]
				count += validation_write(w, indexes[0], new_seq, indexes[2])
				check_length(new_seq)
			}
		} else if len(seq) >= 800 { // If sequence can be padded, pad it and save
			new_seq := pad_seq(seq)
			count += validation_write(w, indexes[0], new_seq, indexes[2])
			check_length(new_seq)
		}
	}

	file.Close()
	return count
}

/*
This function takes TSV files, created from FASTAs, and outputs overlapped 1000bp sequencces.
*/
func overlap_tsv(in string, out string, pad bool) int {
	// Counting CRMs
	count := 0

	// Creates file to write to
	w, err := os.Create(out)
	if err != nil {
		log.Fatal(err)
	}
	// Opens file
	file, err := os.Open(in)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	// Reads file
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		record := scanner.Text()
		indexes := strings.Split(record, "\t")
		name := indexes[0]
		seq := indexes[1]
		length := len(seq)
		num_seqs := int(length / 1000) // Number of divisions in the sample
		remainder := length % 1000     // Remainder of sequence
		// Begin handling sequences
		if length < 800 {
			continue
		} else if pad && length >= 800 && length < 1000 {
			seq = pad_seq(seq)
			count += validation_write(w, name, seq, indexes[2])
		} else if remainder == 0 {
			for i := 0; i < num_seqs; i++ {
				start := 1000 * i
				end := start + 1000
				count += validation_write(w, name, seq[start:end], indexes[2])
			}
		} else if length < 2000 && remainder >= 800 {
			count += validation_write(w, name, seq[0:1000], indexes[2])
			padded_seq := pad_seq(seq[1000:len(seq)])
			count += validation_write(w, name, padded_seq, indexes[2])
		} else if num_seqs < 3 {
			count += validation_write(w, name, seq[0:1000], indexes[2])
			count += validation_write(w, name, seq[len(seq)-1000:len(seq)], indexes[2])
			if remainder >= 800 {
				padded_seq := pad_seq(seq[1000 : 1000+remainder])
				count += validation_write(w, name, padded_seq, indexes[2])
			}
		} else {
			start := 0
			mid_seq := int(num_seqs / 2)      // Finding the middle of the divisions
			total_overlap := 1000 - remainder // 1000 - remainder to get the amount that needs to be spread across divisions
			overlap := int(total_overlap / num_seqs)
			for i := 0; i <= num_seqs; i++ {
				var new_start int
				var new_end int
				if i == 0 {
					new_start = start
					new_end = new_start + 1000
				} else if i == mid_seq {
					start = start - (total_overlap - (overlap * (num_seqs)))
					new_start = 1000*i - overlap*i + start
					new_end = new_start + 1000
				} else {
					new_start = 1000*i - overlap*i + start
					new_end = new_start + 1000
				}
				count += validation_write(w, name, seq[new_start:new_end], indexes[2])
			}
		}
	}

	return count
}

/*
This is used to turn FASTAs in to TSV files with labels.
Checks to make sure sequences aren't on multiple lines and also handles some of the padding.
*/
func parse_fasta(in string, out string, label int, pad bool) int {
	// Declaring variables that will be used later
	// Count to count CRMs, temp to hold each record from FASTA file, next to track if moving to next record, and seq to hold/concat sequences
	count := 0
	var temp = []string{}
	next := false
	seq := ""

	// Creates file to write to
	w, err := os.Create(out)
	if err != nil {
		log.Fatal(err)
	}
	// Writes file with tab delimiter
	writer := csv.NewWriter(w)
	writer.Comma = '\t'
	defer writer.Flush()
	// Opens file
	file, err := os.Open(in)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	// Reads file
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		record := scanner.Text() // Read line as rune, so I can check first index
		if strings.HasPrefix(record, ">") {
			if next {
				if pad && len(seq) < 1000 && len(seq) >= 800 { // Handling padding
					next = false
					seq = strings.Replace(seq, "\n", "", -1)
					seq = pad_seq(seq)
				}
				temp = append(temp, seq)
				temp = append(temp, strconv.Itoa(label)) // Adding label to record
				count++
				if err := writer.Write(temp); err != nil {
					log.Fatalln("error writing record to csv:", err)
				}
				seq = ""
				temp = []string{}
			}
			temp = append(temp, record)
		} else {
			seq += strings.ToUpper(record) // Standardizing all sequences to upper case
			next = true
		}
	}

	// Catching the last record
	if pad && len(seq) < 1000 {
		seq = strings.Replace(seq, "\n", "", -1)
		seq = pad_seq(seq)
	}
	temp = append(temp, seq)
	temp = append(temp, strconv.Itoa(label))
	count++
	if err := writer.Write(temp); err != nil {
		log.Fatalln("error writing record to csv:", err)
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	writer.Flush()
	file.Close()
	return count
}

/*
Handles the reflective padding for the data
*/
func pad_seq(seq string) string {
	remainder := 1000 - len(seq)
	rune_seq := []rune(seq)
	if remainder < 2 {
		seq = string(rune_seq[0]) + seq
	} else {
		half_1 := int(remainder / 2)
		half_2 := remainder - half_1 // Accounting for potential missed base from rounding
		temp_seq_1 := reverse(rune_seq[0:half_1])
		temp_seq_2 := reverse(rune_seq[len(seq)-half_2 : len(seq)])
		seq = temp_seq_1 + seq + temp_seq_2
	}
	return seq
}

/*
Reverses the end sequences for the reflective padding
*/
func reverse(seq []rune) string {
	for i, j := 0, len(seq)-1; i < j; i, j = i+1, j-1 {
		seq[i], seq[j] = seq[j], seq[i]
	}
	return string(seq)
}
