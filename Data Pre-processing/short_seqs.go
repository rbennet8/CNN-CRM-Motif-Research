package main

import (
	"encoding/csv"
	"io"
	"log"
	"strconv"
)

/*
Outputs short sequences.
*/
func parse_short(r *csv.Reader, w io.Writer) int {
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

		if length > 999 || length < 3 {
			continue
		} else {
			count++
			rec := []string{record[0], record[1], record[2]}
			if err := writer.Write(rec); err != nil {
				log.Fatalln("error writing record to csv:", err)
			}
		}
	}
	writer.Flush()
	return count
}
