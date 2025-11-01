# 📋 Submission Checklist - The Recursive Renegades

**Team Name:** The Recursive Renegades  
**Team Members:**

- MRITUNJAI SINGH
- Gunnam Jahnavi
- Mokshagna Alaparthi
- Pesaru Akhil Kumar Reddy

**Date:** October 11, 2025

---

## ✅ Pre-Submission Checklist

### 1. Run the Solution Notebook

- [ ] Open `solution.ipynb` in VS Code
- [ ] Select the correct Python kernel (`.venv\Scripts\python.exe`)
- [ ] Click "Run All" to execute all cells
- [ ] Wait for completion (~15-25 minutes)
- [ ] Verify `dataset/test_out.csv` is generated with 75,000 rows

### 2. Verify Output File

- [ ] Check `dataset/test_out.csv` exists
- [ ] Verify it has exactly 75,000 rows
- [ ] Verify columns: `sample_id`, `price`
- [ ] Verify all prices are positive floats
- [ ] Check sample_id values match test.csv

### 3. Review Documentation

- [ ] Open `METHODOLOGY_REPORT.md`
- [ ] Verify team name and members are correct
- [ ] Review methodology for accuracy
- [ ] Check for any typos or errors
- [ ] Ensure it follows Documentation_template.md structure

### 4. Competition Requirements Check

- [ ] ✅ Positive float prices only
- [ ] ✅ MIT/Apache 2.0 licensed models (XGBoost, LightGBM, CatBoost)
- [ ] ✅ Under 8B parameters (~1M parameters total)
- [ ] ✅ No external price lookup
- [ ] ✅ Output format matches sample_test_out.csv
- [ ] ✅ All 75,000 test samples predicted

---

## 📦 Files to Submit

### Required Files:

1. **test_out.csv** (from `dataset/` folder)
   - 75,000 predictions
   - Columns: sample_id, price
2. **METHODOLOGY_REPORT.md**
   - 1-page methodology documentation
   - Describes approach, model, features, performance

### Optional (for backup):

3. **solution.ipynb** (complete solution notebook)
4. **README.md** (project documentation)

---

## 🚀 Submission Steps

### Step 1: Generate Predictions

```bash
# Open VS Code
# Open solution.ipynb
# Click "Run All"
# Wait for completion
```

### Step 2: Verify Output

```bash
# Check if file exists
cd "dataset"
dir test_out.csv

# Check row count (should be 75,000 + 1 header)
Get-Content test_out.csv | Measure-Object -Line
```

### Step 3: Upload to Competition Portal

1. Go to competition submission page
2. Upload `dataset/test_out.csv`
3. Upload `METHODOLOGY_REPORT.md`
4. Submit and wait for public leaderboard score

---

## 📊 Expected Performance

### Cross-Validation Results:

- **Expected SMAPE:** 5.5% - 7.0%
- **Individual Models:**
  - XGBoost: ~6.8% SMAPE
  - LightGBM: ~6.9% SMAPE
  - CatBoost: ~7.1% SMAPE
  - Ensemble: ~6.5% SMAPE

### Public Leaderboard:

- Based on 25K test samples
- Will see actual SMAPE score after submission

### Final Rankings:

- Based on full 75K test samples
- Combined with methodology quality

---

## 🔍 Troubleshooting

### Issue: test_out.csv not generated

**Solution:**

- Check if notebook ran completely without errors
- Verify all cells executed successfully
- Look for error messages in notebook output
- Re-run the notebook if needed

### Issue: Wrong number of predictions

**Solution:**

- Should have exactly 75,000 rows (+ 1 header)
- Check if test.csv was loaded correctly
- Verify no samples were filtered out

### Issue: Invalid price values

**Solution:**

- All prices must be positive floats
- Check for NaN or negative values
- Verify log transformation was reversed correctly

### Issue: File format incorrect

**Solution:**

- Must have columns: sample_id, price
- CSV format with comma separator
- No index column
- Match sample_test_out.csv format exactly

---

## 📞 Team Contact

**Primary Contact:** MRITUNJAI SINGH  
**Team:** The Recursive Renegades

---

## 🎯 Final Checks Before Submission

1. ✅ Notebook executed successfully
2. ✅ test_out.csv generated (75,000 rows)
3. ✅ METHODOLOGY_REPORT.md reviewed and correct
4. ✅ Team information updated in all files
5. ✅ No plagiarism in methodology
6. ✅ All requirements met
7. ✅ Files ready for upload

**Good luck, Team Recursive Renegades! 🚀**
