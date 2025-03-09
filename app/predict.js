import React, { useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet,TouchableOpacity, Alert, Pressable, ScrollView } from 'react-native';
import { LinearGradient } from "expo-linear-gradient";
import Icon from "react-native-vector-icons/Ionicons";
import { useRouter } from "expo-router";

export default function MaintenancePredictor() {
  const router = useRouter();
  const [kmDriven, setKmDriven] = useState('');
  const [lastServiceDate, setLastServiceDate] = useState('');
  const [result, setResult] = useState(null);

  const handlePredict = async () => {
    if (!kmDriven || !lastServiceDate) {
      Alert.alert('Error', 'Please fill in all fields.');
      return;
    }

    try {
      const response = await fetch(' http://127.0.0.1:8081/predict-maintenance', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          km_driven: parseInt(kmDriven, 10),
          last_service_date: lastServiceDate,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setResult(data);
      } else {
        Alert.alert('Error', 'Failed to get prediction. Please try again.');
      }
    } catch (error) {
      Alert.alert('Error', 'An error occurred. Please check your connection.');
    }
  };

  return (
      <ScrollView contentContainerStyle={styles.scrollContainer}>
        <Pressable style={styles.backButton} onPress={() => router.push("/home")}>
          <Icon name="arrow-back" size={25} color="black" />
          <Text style={styles.backText}>Back</Text>
        </Pressable>
        
        <Text style={styles.title}>Bike Maintenance Predictor</Text>

        <Text style={styles.label}>Kilometers Driven:</Text>
        <TextInput
          style={styles.input}
          keyboardType="numeric"
          value={kmDriven}
          onChangeText={setKmDriven}
          placeholder="Enter kilometers driven"
          placeholderTextColor="#bbb"
        />

        <Text style={styles.label}>Last Service Date (YYYY-MM-DD):</Text>
        <TextInput
          style={styles.input}
          value={lastServiceDate}
          onChangeText={setLastServiceDate}
          placeholder="Enter last service date"
          placeholderTextColor="#bbb"
        />

   
<TouchableOpacity
          onPress={handlePredict}
          style={[styles.button, styles.registerButton]}
        >
          <Text style={styles.buttonText}>Predict Maintenance</Text>
        </TouchableOpacity>
        {result && (
          <View style={styles.result}>
            <Text style={styles.resultText}>Battery Score: {result.battery_score}</Text>
            <Text style={styles.resultText}>Maintenance Score: {result.maintenance_score}</Text>
            <Text style={styles.resultText}>
              Replacement Needed: {result.replacement_needed ? 'Yes' : 'No'}
            </Text>
            <Text style={styles.resultText}>
              Maintenance Needed: {result.maintenance_needed ? 'Yes' : 'No'}
            </Text>
          </View>
        )}
      </ScrollView>
  );
}
const styles = StyleSheet.create({
  scrollContainer: {
    flexGrow: 1,
    paddingHorizontal: 24,
    paddingVertical: 40,
    backgroundColor: "#f8f9fa",
  },
  backButton: {
    position: "absolute",
    top: 40,
    left: 20,
    flexDirection: "row",
    alignItems: "center",
    zIndex: 10,
  },
  backText: {
    marginLeft: 5,
    fontSize: 20,
    color: "#000", 
  },
  title: {
    fontSize: 26,
    fontWeight: "bold",
    color: "#264653",
    textAlign: "center",
    marginBottom: 32,  // Increased spacing under the title
    marginTop: 60,
  },
  label: {
    fontSize: 20,
    color: "#1e40af",
    marginBottom: 12,  // Spaced out from the input below
  },
  input: {
    borderWidth: 1,
    borderColor: "#ced4da",
    borderRadius: 8,
    fontSize: 18,
    padding: 15,
    color: "#212529",
    backgroundColor: "#ffffff",
    marginBottom: 24,  // Consistent spacing between inputs
  },
  button: {
    backgroundColor: "#005f73",
    paddingHorizontal: 32,
    paddingVertical: 16,  // Slightly larger padding for a better touch target
    borderRadius: 16,
    marginBottom: 24,  // Space below the button
  },
  buttonText: {
    textAlign: "center",
    color: "#ffffff",
    fontWeight: "bold",
    fontSize: 18,
  },
  result: {
    marginTop: 32,  // Separate result section from the form
    padding: 16,
    borderWidth: 1,
    borderColor: "#94d2bd",
    borderRadius: 8,
    backgroundColor: "#e0f5f3",
  },
  resultText: {
    textAlign: "center",
    fontSize: 20,
    color: "#264653",
    marginBottom: 8,
  },
});
