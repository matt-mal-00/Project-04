# ==========================================
# STEP 6: EVALUATION & ROC CURVE
# ==========================================
plt.figure(figsize=(10, 8))

for name, model in trained_models.items():
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Print Classification Report
    print(f"\n--- Performance Metrics: {name} ---")
    print(classification_report(y_test, y_pred))
    
    # Calculate AUC and Plot ROC
    auc_score = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')

# Finalize ROC Curve Plot
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# Save 300 DPI image for manuscript
plt.savefig('roc_curve_300dpi.png', dpi=300, bbox_inches='tight')
plt.show()