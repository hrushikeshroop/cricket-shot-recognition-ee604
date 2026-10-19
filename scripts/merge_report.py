"""Append the complete code PDF to a freshly exported Overleaf manuscript."""
import argparse
from pathlib import Path
import pymupdf

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('manuscript',type=Path)
    parser.add_argument('appendix',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.output.resolve() in {args.manuscript.resolve(),args.appendix.resolve()}:
        raise ValueError('Choose a new output path; keep the input PDFs intact.')
    with pymupdf.open(args.manuscript) as paper,pymupdf.open(args.appendix) as code:
        paper.insert_pdf(code)
        paper.save(args.output,garbage=4,deflate=True)
        print('Saved',len(paper),'pages to',args.output)

if __name__=='__main__':main()
