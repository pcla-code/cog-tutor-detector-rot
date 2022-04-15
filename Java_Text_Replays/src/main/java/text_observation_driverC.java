import java.io.*;
import java.util.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

public class text_observation_driverC{

    String action_data_file = "unused_clips4.txt";
	String processed_data_file = "";
	//53 students in brockton_spring_2021_MATHia_data_TST_pilot.tsv
	String raw_mathia_file ="Mathia_Log.tsv";

   
    String outfile_nothi = "observations";
    String repeatfile = "toredo";
    
    public String coder_id = "BOB";
    
    StreamTokenizer st_;
    FileWriter fw_;

	int numClips = -1;
    public int totalex = 0;

    public boolean anonymized = true; // 2003 not anonymized. 2004, 2005 anonymized.

    public boolean repeatOldFirst = true; // for second coder, etc.

    public void text_observationdriverC(){}

	//Read in raw data from Mathia and prep for coding
    public void data_prep(){
		try (
				Reader reader = Files.newBufferedReader(Paths.get(raw_mathia_file));
				CSVParser csvParser = new CSVParser(reader, CSVFormat.TDF
						.withFirstRecordAsHeader()
						.withIgnoreHeaderCase()
						.withTrim());
				CSVPrinter csvPrinter = new CSVPrinter(new FileWriter(processed_data_file),CSVFormat.DEFAULT
						.withHeader("Student","Assess","Cell","Celltype","answer","prod","pknow","time","helpintermedtime","numsteps"));
		){
			int i =0;
			String student = "";
			double timeStamp = 0;
			double time = 0.0;
			String cell = "";
			String answer = "";
			String assess = "";
			String prod = "";
			String pknow = "";
			String numsteps = "";
			String cellType = "";
			ArrayList<String[]> records = new ArrayList<String[]>();
			for (CSVRecord csvRecord: csvParser)
			{
				cellType = csvRecord.get(4);
				if(cellType.startsWith("pre_launch"))
				{
					continue;
				}
				double nextTimeStamp = Double.parseDouble(csvRecord.get(2));
				String nextStudent=csvRecord.get(0);
				if(!nextStudent.equals(student))
					time = 0.0;
				else {
					time = ((nextTimeStamp-timeStamp) * 0.001);
				}
				if(i != 0)
					csvPrinter.printRecord(new String[]{student, assess, cell, cellType, answer, prod, pknow, String.valueOf(time), "-1", numsteps});

				student = csvRecord.get(0);
				timeStamp = Double.parseDouble(csvRecord.get(2));
				cell = csvRecord.get(5);
				answer = csvRecord.get(8);
				assess = csvRecord.get(9);
				if (!csvRecord.get(12).equals("") || csvRecord.get(7).equals("Done"))
				{
					prod = csvRecord.get(12);
					pknow = csvRecord.get(14);
				}

				numsteps = csvRecord.get(10);
				i++;
			}
			time = 0.0;
			csvPrinter.printRecord(new String[]{student, assess, cell, cellType, answer, prod, pknow, String.valueOf(time), "-1", numsteps});




		}
		catch (IOException e){
			e.printStackTrace();
		}

	}



	public StreamTokenizer create_tokenizer(){
	
	try{
	    return new StreamTokenizer(new FileReader(action_data_file));
	}
	catch (FileNotFoundException fnfe){
	    fnfe.printStackTrace();
	}
	return null;
    }
    
    public void getLengthOfExisting(){
	try{
	    StreamTokenizer st = new StreamTokenizer(new FileReader(outfile_nothi+coder_id));
	    int tt= StreamTokenizer.TT_NUMBER; totalex = 0;
	    while (tt!= StreamTokenizer.TT_EOF){
		tt = st.nextToken();
		if (tt == StreamTokenizer.TT_EOL)
		    totalex++;
		if ((tt == StreamTokenizer.TT_WORD)&&(st.sval.equals(coder_id)))
		    totalex++;
	    }
	}
	catch (Exception fnfe){
	}
    }

    public int getRepeatClip(int goalclip){
	try{
	    StreamTokenizer st = new StreamTokenizer(new FileReader(repeatfile+coder_id));
	    int tt= StreamTokenizer.TT_NUMBER; int curclipnum = 0; int toret = -1;
	    while (tt!= StreamTokenizer.TT_EOF){
			tt = st.nextToken();
			if (tt == StreamTokenizer.TT_EOL)
				curclipnum++;
			if ((tt == StreamTokenizer.TT_WORD)&&((st.sval.equals("N"))||(st.sval.equals("G"))))
				curclipnum++;
			if (curclipnum == goalclip)
				return toret;
			if (tt == StreamTokenizer.TT_NUMBER)
				toret = (new Double(st.nval)).intValue();
	    }
	}
	catch (Exception fnfe){
	    return -1;
	}
	return -1;
    }

    public FileWriter out_tokenizer(){
	try{
	    return new FileWriter(outfile_nothi+coder_id,true);
	}
	catch (Exception fnfe){
	    fnfe.printStackTrace();
	}
	return null;
    }

    public void append_record (String toadd){
	try{
	    fw_ = out_tokenizer();
	    fw_.write(toadd);
	    fw_.flush();
	    fw_.close();
	}
	
	catch (Exception fnfe){
	    fnfe.printStackTrace();
	}
    }
    
    public String getIdentity (){
		return javax.swing.JOptionPane.showInputDialog("Input Coder ID");
    }
	public String getFileName()
	{
		return javax.swing.JOptionPane.showInputDialog("Enter Input File Name (File must be in the same folder as this program):");
	}
	public String get_output_fileName()
	{
		return javax.swing.JOptionPane.showInputDialog("Enter Output File Name (should end in .txt):");
	}
	public Object query_task(){
		Object[] options = {"Prepare Data File", "Code Data"};
		return javax.swing.JOptionPane.showOptionDialog(null,"Choose a function:","Prep or Code",javax.swing.JOptionPane.DEFAULT_OPTION, javax.swing.JOptionPane.WARNING_MESSAGE, null, options, options[1]);
	}

    public Object query_gaming(String title, String msg){
	Object[] options = { "GAMING", "NOT GAMING", "BAD CLIP" };
	return javax.swing.JOptionPane.showOptionDialog(null, msg, title, javax.swing.JOptionPane.DEFAULT_OPTION, javax.swing.JOptionPane.WARNING_MESSAGE, null, options, options[1]);
    }

    public Object query_gaming_more(String title, String msg){
	Object[] options = { "GAMING", "NOT GAMING", "BAD CLIP", "MORE" };
	return javax.swing.JOptionPane.showOptionDialog(null, msg, title, javax.swing.JOptionPane.DEFAULT_OPTION, javax.swing.JOptionPane.WARNING_MESSAGE, null, options, options[1]);
    }

    String student_[] = new String[523593];
    String assess_[] = new String[523593];
    String cell_[] = new String[523593];
    String celltype_[] = new String[523593];
    String prod_[] = new String[523593];
    String answer_[] = new String[523593];
    double pknow_[] = new double[523593];
    double time_[] = new double[523593];
    double numsteps_[] = new double[523593];
    double helpintermedtime_[] = new double[523593];
	
    // humaninterpretable has headers and prod names, etc etc
    public void readInActions(){

		st_ = create_tokenizer();

		st_.wordChars(32,32);
		//	fw_ = out_tokenizer();
		boolean quitnow = false;

		int num = -1; String namea = ""; String assess = "";
		String cell = ""; String celltype = ""; String prod = "";
		String answer = "";
		double pknow = 0; double time = 0; double pknowretro = -1;
		double prevpknow = 0;

		String help = "0"; double numsteps = 0; double helpintermedtime = 0;

		String curstu = ""; String punchange = "0";

		while (!quitnow){
			namea = ""; assess = "";
			cell = ""; celltype = ""; prod = "";
			pknow = 0; time = 0; answer = "";
			helpintermedtime = 0; numsteps = 0;
			pknowretro = 0;

			try{
			int tt = st_.nextToken();
			if (tt == StreamTokenizer.TT_EOF)
				quitnow=true;
			else{
			num+=1;

			while (st_.sval == null){
				tt = st_.nextToken();
				if (tt == StreamTokenizer.TT_EOF)
				throw new Exception("Reached EOF.");
			}

			namea = st_.sval;

			st_.nextToken();
			while (st_.sval == null){
				tt= st_.nextToken();
				if (tt == StreamTokenizer.TT_EOF)
				throw new Exception("Reached EOF.");
			}
			assess = st_.sval;

			st_.nextToken();
			while (st_.sval == null){
				tt = st_.nextToken();
				if (tt == StreamTokenizer.TT_EOF)
				throw new Exception("Reached EOF.");
			}
			cell = st_.sval;

			st_.nextToken();
			while (st_.sval == null){
				tt= st_.nextToken();
				if (tt == StreamTokenizer.TT_EOF)
				throw new Exception("Reached EOF.");
			}
			celltype = st_.sval;

				st_.nextToken();
				while (st_.sval == null){
				tt= st_.nextToken();
				if (tt == StreamTokenizer.TT_EOF)
					throw new Exception("Reached EOF.");
				}
				answer = st_.sval;
				//}
			//	}


			st_.nextToken();
			while (st_.sval == null){
				tt = st_.nextToken();
				if (tt == StreamTokenizer.TT_EOF)
				throw new Exception("Reached EOF.");
			}
			prod = st_.sval;

			tt = st_.nextToken();
			while (tt != StreamTokenizer.TT_NUMBER)
				tt = st_.nextToken();
			pknow = st_.nval;
			pknowretro = pknow;
			if (pknow!=-1)
				prevpknow=pknow;
			else
				pknow=prevpknow;

				tt = st_.nextToken();
			while (tt != StreamTokenizer.TT_NUMBER)
				tt = st_.nextToken();
			time = st_.nval;

			tt = st_.nextToken();
			while (tt != StreamTokenizer.TT_NUMBER)
				tt = st_.nextToken();
			helpintermedtime = st_.nval;

			tt = st_.nextToken();
			while (tt != StreamTokenizer.TT_NUMBER)
				tt = st_.nextToken();
			numsteps = st_.nval;

			if (tt == StreamTokenizer.TT_EOF)
				quitnow=true;

			pknow = pknow * 100;
			long temp = Math.round(pknow);
			pknow = (new Long(temp)).doubleValue();
			pknow = pknow / 100;

			if (celltype.startsWith("CHECK"))
				answer="CHECK";

			answer = answer.replaceAll("ZYX"," ");
			student_[num] = namea;
			assess_[num] = assess;
			cell_[num] = cell;
			celltype_[num] = celltype;
			prod_[num] = prod;
			answer_[num] = answer;
			pknow_[num] = pknow;
			time_[num] = time;
			numsteps_[num] = numsteps;
			helpintermedtime_[num] = helpintermedtime;

			}

			}catch(Exception e){ quitnow=true;}
		}
    }


	public void read_all_actions()
	{
		try (
				Reader reader = Files.newBufferedReader(Paths.get(action_data_file));
				CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT
						.withFirstRecordAsHeader()
						.withIgnoreHeaderCase()
						.withTrim());
		) {
			int i = 0;
			for (CSVRecord csvRecord: csvParser) {
				student_[i] = csvRecord.get(0);
				assess_[i] = csvRecord.get(1);
				cell_[i] = csvRecord.get(2);
				celltype_[i] = csvRecord.get(3);
				answer_[i] = csvRecord.get(4);
				prod_[i] = csvRecord.get(5);
				pknow_[i] = (csvRecord.get(6).equals("")) ? -1  : Double.parseDouble(csvRecord.get(6));
				time_[i] = Double.parseDouble(csvRecord.get(7));
				helpintermedtime_[i] = Double.parseDouble(csvRecord.get(8));
				numsteps_[i] = Double.parseDouble(csvRecord.get(9));
				i++;
			}
			numClips = i;
		}
		catch (IOException e){
			e.printStackTrace();
		}

	}

	public void write_clips(int numClips, int numStudents)
	{
		try (
				Reader reader = Files.newBufferedReader(Paths.get(action_data_file));
				CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT
						.withFirstRecordAsHeader()
						.withIgnoreHeaderCase()
						.withTrim());
				CSVPrinter csvPrinter = new CSVPrinter(new FileWriter(processed_data_file),CSVFormat.DEFAULT
						.withHeader("Student","Assess","Cell","Celltype","answer","prod","pknow","time","helpintermedtime","numsteps"));
				CSVPrinter csvPrinter2 = new CSVPrinter(new FileWriter("unused_clips5.txt"),CSVFormat.DEFAULT
						.withHeader("Student","Assess","Cell","Celltype","answer","prod","pknow","time","helpintermedtime","numsteps"));

		){
			int i =0;
			int stu = 0;
			int count = numClips/numStudents;
			if(numClips%numStudents-1 >= stu)
			{
				count++;
			}
			String student = "";
			double time = 0.0;
			double totalTime =0.0;
			String cell = "";
			String answer = "";
			String assess = "";
			String prod = "";
			String pknow = "";
			String numsteps = "";
			String cellType = "";
			ArrayList<String[]> records = new ArrayList<String[]>();
			for (CSVRecord csvRecord: csvParser)
			{
				if(!student.equals(csvRecord.get(0))){
					if(count>0)
					{
						stu -= count;
					}
					stu++;
					count = numClips/numStudents;
					if(numClips%numStudents-1 >= stu)
					{
						count++;
					}
				}
				if (count>0)
				{
					student = csvRecord.get(0);
					assess = csvRecord.get(1);
					cell = csvRecord.get(2);
					cellType = csvRecord.get(3);
					answer = csvRecord.get(4);
					prod = csvRecord.get(5);
					pknow = csvRecord.get(6);
					time = Double.parseDouble(csvRecord.get(7));
					totalTime += time;
					numsteps = csvRecord.get(9);
					csvPrinter.printRecord(new String[]{student, assess, cell, cellType, answer, prod, pknow, String.valueOf(time), "-1", numsteps});
					if(totalTime >= 20.0){
						count--;
						totalTime =0.0;
					}
				}
				else if (count == 0)
				{
					student = csvRecord.get(0);
					assess = csvRecord.get(1);
					cell = csvRecord.get(2);
					cellType = csvRecord.get(3);
					answer = csvRecord.get(4);
					prod = csvRecord.get(5);
					pknow = csvRecord.get(6);
					time = Double.parseDouble(csvRecord.get(7));
					numsteps = csvRecord.get(9);
					csvPrinter2.printRecord(new String[]{student, assess, cell, cellType, answer, prod, pknow, String.valueOf(time), "-1", numsteps});
				}
			}


		}
		catch (IOException e){
			e.printStackTrace();
		}


	}
    public int displayClip (int clipID){
		return displayClip(clipID, clipID, 0.0);
    }


    public int displayClip (int clipID, int originalID, double totaltime){
    
	String clip_text = ""; String title = "";

	title = "Observation " + (new Integer(totalex+1)).toString() + " for coder " + coder_id + ": Clip " +  (new Integer(clipID)).toString() + "\n";	
	int curnum = clipID; int actioncount= 0;
   
	while ((totaltime < 20)&&(actioncount<8)){
	    actioncount++;
	    clip_text = clip_text + "Time " + (new Double(totaltime)).toString() + ":\n" ;

	    if (!(assess_[curnum].equals("HELP"))){
		clip_text = clip_text + "Entered " + answer_[curnum] + " into " + cell_[curnum] + " (" + celltype_[curnum] + ")\n";
		
		clip_text = clip_text + "Assessment: " + assess_[curnum] + "\n";
		
		clip_text = clip_text + "Production: " + prod_[curnum]; 

		if (pknow_[curnum]==-1)
		    clip_text = clip_text + " (pknow: UNKNOWN)" + "\n";
		else{
		    clip_text = clip_text + " (pknow: " + (new Double(pknow_[curnum])).toString() + ")\n" ;
		}
		clip_text = clip_text + "\n"; 
	    }else{
		clip_text = clip_text + "Requested help on production " + "\n"; 
		clip_text = clip_text + prod_[curnum]; 
		
		if (pknow_[curnum]==-1)
		    clip_text = clip_text + " (pknow: UNKNOWN)" + "\n";
		else{
		    clip_text = clip_text + " (pknow: " + (new Double(pknow_[curnum])).toString() + ")\n" ;
		}

		if (numsteps_[curnum]==-1)
		    clip_text = clip_text + "Read UNKNOWN steps." + "\n";
		else
		    clip_text = clip_text + "Read " + (new Double(numsteps_[curnum]+1)).toString() + " steps." + "\n";

		clip_text = clip_text + "\n"; 
	    }
	    totaltime += time_[curnum];
	    curnum++;
	    if ((actioncount==8)&&(totaltime<20))
		clip_text = clip_text + "More....";
	}
	//System.out.print(clip_text);
	Object result_o = null;
	if ((actioncount == 8)&&(totaltime<20)) 
	    result_o = query_gaming_more(title, clip_text);
	else
	    result_o = query_gaming(title, clip_text);
	int result = ((Integer)result_o).intValue();
	String result_s = "?";
	if (result==0)
	    result_s = "G";
	if (result==1)
	    result_s = "N";
	if (result==2)
	    result_s = "?";
	if (result<3){
	    String toadd = coder_id + " " + (new Integer(originalID)).toString() + " " + result_s + "\n";
	    append_record(toadd);
	}
	else
	    return displayClip(curnum, clipID, totaltime);
	return curnum;
}

    public static void main (String args[]) {
		text_observation_driverC mdd = new text_observation_driverC();

		mdd.processed_data_file = "./" + mdd.getFileName();
		mdd.write_clips(60, 53);

		/*
		Object option_o = mdd.query_task();
		int choice = ((Integer)option_o).intValue();
		if (choice==0) {
			mdd.raw_mathia_file = "./"+mdd.getFileName();
			mdd.processed_data_file = "./"+mdd.get_output_fileName();
			mdd.data_prep();
		}
		if (choice==1) {
			mdd.action_data_file = "./"+mdd.getFileName();

			int clipID = -1;

			//Get coder id, typically the initials of the coder
			mdd.coder_id = mdd.getIdentity();

			//update totalex with the total number of clips coded by this user, as totaled in the observations file
			mdd.getLengthOfExisting();

			//Read the entire data file into local array storage
			mdd.read_all_actions();

			Random gen = new Random();
			clipID = 0;
			while (clipID < mdd.numClips) {

				clipID = mdd.displayClip(clipID);

				mdd.getLengthOfExisting();
			}
		}

		 */

	}

}
