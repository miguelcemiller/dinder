# Generated by Django 4.0.1 on 2022-02-03 02:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('application', '0004_alter_profile_username'),
    ]

    operations = [
        migrations.AddField(
            model_name='profile',
            name='dog_breed',
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
    ]