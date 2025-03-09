import { useRouter } from "expo-router";
import { View, Text, Pressable, StyleSheet } from "react-native";
import Icon from "react-native-vector-icons/Ionicons";

export default function Home() {
  const router = useRouter();

  return (
    <View style={styles.container}>
      <Pressable style={styles.backButton} onPress={() => router.push("/")}>
        <Icon name="arrow-back" size={24} color="#1E3A8A" />
        <Text style={styles.backText}>Back</Text>
      </Pressable>

      <Text style={styles.heading}>Risk Management Made Simple</Text>

      <View style={styles.dashboard}>
        <Pressable style={styles.card} onPress={() => router.push("/score")}>
          <Icon name="stats-chart" size={32} color="#005f73" style={styles.cardIcon} />
          <Text style={styles.cardTitle}>Know Your Risk</Text>
          <Text style={styles.cardDescription}>
            Check your current risk level and get safety suggestions.
          </Text>
        </Pressable>

        <Pressable style={styles.card} onPress={() => router.push("/predict")}>
          <Icon name="construct" size={32} color="#059669" style={styles.cardIcon} />
          <Text style={styles.cardTitle}>Predict Maintenance</Text>
          <Text style={styles.cardDescription}>
            Get maintenance predictions to keep your bike in top shape.
          </Text>
        </Pressable>

        <Pressable style={styles.card} onPress={() => router.push("/sos")}>
          <Icon name="alert-circle" size={32} color="#dc2626" style={styles.cardIcon} />
          <Text style={styles.cardTitle}>SOS</Text>
          <Text style={styles.cardDescription}>Send an emergency alert quickly.</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#ffffff", // Pure white background
    paddingTop: 60,
    paddingHorizontal: 20,
  },
  backButton: {
    position: "absolute",
    top: 40,
    left: 20,
    flexDirection: "row",
    alignItems: "center",
  },
  backText: {
    marginLeft: 5,
    fontSize: 16,
    color: "#1E3A8A", // Blue color for the back text
  },
  heading: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#1E3A8A", // Deep blue for the heading
    textAlign: "center",
    marginBottom: 20,
  },
  dashboard: {
    flex: 1,
    justifyContent: "center",
  },
  card: {
    backgroundColor: "#ffffff",
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: "#1E90FF",
    elevation: 3, // Adds subtle shadow for Android
    shadowColor: "#000", // Adds shadow for iOS
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    alignItems: "center",
  },
  cardIcon: {
    marginBottom: 10,
  },
  cardTitle: {
    fontSize: 20,
    fontWeight: "bold",
    color: "#2563EB",
    marginBottom: 8,
  },
  cardDescription: {
    fontSize: 16,
    color: "#374151", 
    textAlign: "center",
  },
});
