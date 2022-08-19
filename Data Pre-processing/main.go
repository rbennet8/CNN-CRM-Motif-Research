package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
)

/*
Our data was in NAME, START, STOP type CSV format, so I used those start and stop values to calculate sequnce lengths.
	Then, I used Bedtools to get the FASTA files for each sequence.
The data used for validation was in FASTA format, so I had to read those and parse them as I went.
	There was no Bedtools use or anything.
That's why there are so many different types of functions for different sequence handling, because the starting points were different.
*/

func main() {
	// Takes two bool values
	// 0 - True if parsing short sequences
	// 1 - True if overlapping in long sequences
	//choices := [2]bool{false, false}
	//parsing_lengths(choices)

	parsing_fastas()
}

/*
This function reads in CRM and non-CRM data and calls appropriate methods, depending on booleans.
*/
func parsing_lengths(short_overlap [2]bool) {
	// crm_file, err := os.Open("test.txt")
	crm_file, err := os.Open("MouseCRMs.txt")
	if err != nil {
		log.Fatal(err)
	}
	/* Commenting out this block, because I'm only processing CRM data
	// noncrm_file, err := os.Open("test.txt")
	noncrm_file, err := os.Open("MouseNonCRMs.txt")
	if err != nil {
		log.Fatal(err)
	}
	*/

	crm_reader := csv.NewReader(crm_file)
	crm_reader.Comma = '\t'
	/* Commenting out this block, because I'm only processing CRM data
	noncrm_reader := csv.NewReader(noncrm_file)
	noncrm_reader.Comma = '\t'
	*/

	var crm_filename string
	/* Commenting out this block, because I'm only processing CRM data
	var noncrm_filename string
	*/
	if short_overlap[0] {
		crm_filename = "MouseShortCRMs.txt"
		//noncrm_filename = "MouseShortNonCRMs.txt"
	} else if short_overlap[1] {
		crm_filename = "MouseCRMsOverlap.txt"
		//noncrm_filename = "MouseNonCRMsOverlap.txt"
	} else {
		crm_filename = "MouseCRMsDropped.txt"
		//noncrm_filename = "MouseNonCRMsDropped.txt"
	}

	crm_writer, err := os.Create(crm_filename)
	if err != nil {
		log.Fatal(err)
	}
	/* Commenting out this block, because I'm only processing CRM data
	noncrm_writer, err := os.Create(noncrm_filename)
	if err != nil {
		log.Fatal(err)
	}
	*/

	if short_overlap[0] {
		crm_count := parse_short(crm_reader, crm_writer)
		fmt.Println("CRM Entries: ", crm_count)
		crm_writer.Close()
		/* Commenting out this block, because I'm only processing CRM data
		noncrm_count := parse_short(noncrm_reader, noncrm_writer)
		fmt.Println("Non-CRM Entries: ", noncrm_count)
		noncrm_writer.Close()
		*/
	} else if short_overlap[1] {
		crm_count := long_overlap(crm_reader, crm_writer)
		fmt.Println("CRM Entries: ", crm_count)
		crm_writer.Close()
		/* Commenting out this block, because I'm only processing CRM data
		noncrm_count := long_overlap(noncrm_reader, noncrm_writer)
		fmt.Println("Non-CRM Entries: ", noncrm_count)
		noncrm_writer.Close()
		*/
	} else {
		crm_count := long_drop(crm_reader, crm_writer)
		fmt.Println("CRM Entries: ", crm_count)
		crm_writer.Close()
		/* Commenting out this block, because I'm only processing CRM data
		noncrm_count := long_drop(noncrm_reader, noncrm_writer)
		fmt.Println("Non-CRM Entries: ", noncrm_count)
		noncrm_writer.Close()
		*/
	}
}

/*
This function handles the appropriate calls for handle various FASTA files.
*/
func parsing_fastas() {
	// Declaring these here, so I can comment and uncomment commands as needed
	var crm_in string
	var crm_out string
	var crm_count int

	// parse_fasta parameters are - file in, file out, label, and padding.
	//crm_in := "Test.txt"
	//crm_out := "TestTSV.txt"
	//crm_count := parse_fasta(crm_in, crm_out, 0, false)
	//fmt.Println(crm_count)

	// overlap_tsv parameters are - file in, file out, and padding
	crm_in = "HumanCRMValidationFastaTSV.txt"
	crm_out = "HumanCRMValidationDataOverlapped.txt"
	crm_count = overlap_tsv(crm_in, crm_out, true)
	fmt.Println(crm_count)

	/*
		// FASTA files from Bedtools
		crm_fastas := [3]string{"MouseCRMDroppedFasta.txt", "MouseCRMOverlapFasta.txt", "MouseShortCRMFasta.txt"}
		noncrm_fastas := [3]string{"MouseNonCRMDroppedFasta.txt", "MouseNonCRMOverlapFasta.txt", "MouseShortNonCRMFasta.txt"}

		// Output filenames
		crm_filenames := [3]string{"MouseDroppedCRMSeqs.txt", "MouseOverlappedCRMSeqs.txt", "MouseShortCRMSeqs.txt"}
		noncrm_filenames := [3]string{"MouseDroppedNonCRMSeqs.txt", "MouseOverlappedNonCRMSeqs.txt", "MouseShortNonCRMSeqs.txt"}

		for i := 0; i < 3; i++ {
			crm_count, noncrm_count := 0, 0
			if i == 2 {
				crm_count += parse_fasta(crm_fastas[i], crm_filenames[i], 0, false)
				noncrm_count += parse_fasta(noncrm_fastas[i], noncrm_filenames[i], 1, false)
				fmt.Println(crm_filenames[i], " - ", crm_count)
				fmt.Println(noncrm_filenames[i], " - ", noncrm_count)
			} else {
				crm_count += parse_fasta(crm_fastas[i], crm_filenames[i], 0, true)
				noncrm_count += parse_fasta(noncrm_fastas[i], noncrm_filenames[i], 1, true)
				fmt.Println(crm_filenames[i], " - ", crm_count)
				fmt.Println(noncrm_filenames[i], " - ", noncrm_count)
			}
		}
	*/
}
